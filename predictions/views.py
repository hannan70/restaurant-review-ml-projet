from django.shortcuts import render
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import os
import pickle
from .models import Feedback 
from collections import Counter

# Download NLTK data (this should be done only once during server startup)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up preprocessing parameters
voc_size = 10000
max_length = 500
lemmatize = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
important_words = {"not", "no", "nor", "never"}
final_stopwords = stop_words - important_words

# Load model and tokenizer
# Load model and tokenizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'ml/simple_rnn.h5')
tokenizer_path = os.path.join(BASE_DIR, 'ml/tokenizer.pkl')

model = load_model(model_path, compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess input - defined at module level
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = word_tokenize(review)
    review = [lemmatize.lemmatize(word) for word in review if word not in final_stopwords]
    cleaned_text = " ".join(review).strip()
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_length)
    return padded

# Prediction function - defined at module level
def predict_sentiment(review):
    processed_input = preprocess_text(review)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.6 else 'Negative'
    return sentiment, prediction[0][0]

# Create your views here
def home_page(request):
    sentiment = None
    score_percentage = None
    score = None
    total_positive_review = 0
    total_negative_review = 0

    if request.method == 'POST':
        review_text = request.POST.get("review", "").strip()
        if review_text:
            sentiment, score = predict_sentiment(review_text)
            score_percentage = f"{score * 100:.2f}%"

            # save data into database
            Feedback.objects.create(
                review_text = review_text,
                sentiment = sentiment
            ) 
        else:
            sentiment = "Invalid"
            score_percentage = "0.00%"
        
        context = {
            "sentiment": sentiment,
            "score": score_percentage,
            "total_positive_review": total_positive_review,
            "total_negative_review": total_negative_review,
        }

    # read all feedbacks
    feedbacks = Feedback.objects.all()
    # count sentiment
    all_sentiments = [ feedback.sentiment for feedback in feedbacks]
    sentiment_count = Counter(all_sentiments)
    total_positive_review = sentiment_count.get('Positive', 0)
    total_negative_review = sentiment_count.get('Negative', 0)


    context = {
        "sentiment": sentiment,
        "score": score_percentage,
        "total_positive_review": total_positive_review,
        "total_negative_review": total_negative_review,
    }
        
    return render(request, "index.html", context)