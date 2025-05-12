from django.shortcuts import render
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bnltk.tokenize import Tokenizers
from langdetect import detect, LangDetectException
import re
import nltk
import os
import pickle
from .models import Feedback 
from collections import Counter
import string
import codecs
from gtts import gTTS
import time

# Download NLTK data (this should be done only once during server startup)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
punctuation_marks = set(string.punctuation)

# Set up preprocessing parameters
voc_size = 10000
max_length_bn = 149
max_length = 500
lemmatize = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
important_words = {"not", "no", "nor", "never"}
final_stopwords = stop_words - important_words
stopword = codecs.open("./ml/dataset/stopwords-bn.txt", 'r', encoding='utf-8').read().split()

 

# Load model and tokenizer for english
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'ml/simple_rnn.h5')
tokenizer_path = os.path.join(BASE_DIR, 'ml/tokenizer.pkl')

# Load model and tokenizer for bangla
model_path_bn = os.path.join(BASE_DIR, 'ml/bangla_lstm.h5')
tokenizer_path_bn = os.path.join(BASE_DIR, 'ml/bangla_tokenizer.pkl')


# Load and compile model for english
model = load_model(model_path, compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load and compile model for bangla
model_bn = load_model(model_path_bn, compile=False)
model_bn.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
with open(tokenizer_path_bn, 'rb') as file:
    tokenizer_bn = pickle.load(file)


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

# # Text preprocess for bangla 
def preprocess_text_for_bangla(text):
    reviews = re.sub(r'\b[a-zA-Z]+\b', '', text)
    # tokenize
    words = Tokenizers.bn_word_tokenizer(reviews)
    # remove punctuation marks
    reviews = [word for word in words if word not in punctuation_marks]
    # remove bangla stopwords
    reviews = [word for word in reviews if word not in stopword]
    cleaned_text = " ".join(reviews).strip()
    encoded_review = tokenizer_bn.texts_to_sequences([cleaned_text])
    padded_review = pad_sequences(encoded_review, maxlen=max_length_bn)
    return padded_review

# Language detect 
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"

# Prediction function - defined at module level
def predict_sentiment(review):
    language = detect_language(review)

    if language == "bn":
        processed_input = preprocess_text_for_bangla(review)
        prediction = model_bn.predict(processed_input)
    elif language == "en":
        processed_input = preprocess_text(review)
        prediction = model.predict(processed_input)
    else:
        return "Unknown Language"

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Create your views here
def home_page(request):
    sentiment = None
    score_percentage = "0.00%"
    score = None
    total_positive_review = 0
    total_negative_review = 0
    audio_file = None
    input_value = ""
    
    if request.method == 'POST':
        review_text = request.POST.get("review", "").strip()
        action = request.POST.get("action")
        input_value = review_text

        if action =='predict' and review_text:
            sentiment, score = predict_sentiment(review_text)
            score_percentage = f"{score * 100:.2f}%"

            # save data into database
            Feedback.objects.create(
                review_text = review_text,
                sentiment = sentiment
            ) 
        elif action == 'tts' and review_text:

            filename = f"output_{int(time.time())}.mp3"
            audio_path = os.path.join("static", filename)
            language = detect_language(review_text)
            
            # Convert text to speech
            tts = gTTS(text=review_text, lang=language, slow=False)
            tts.save(audio_path)

            # Return relative path to static folder
            audio_file = f"static/{filename}"

            # Clean up old audio files to avoid clutter
            for f in os.listdir("static"):
                if f.startswith("output_") and f != filename:
                    os.remove(os.path.join("static", f))


        else:
            sentiment = "Invalid"
            score_percentage = "0.00%"
        
        context = {
            "sentiment": sentiment,
            "score": score_percentage,
            "total_positive_review": total_positive_review,
            "total_negative_review": total_negative_review,
            "audio": audio_file,
            "input_value": input_value
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
        "audio": audio_file,
        "input_value": input_value
    }
        
    return render(request, "index.html", context)