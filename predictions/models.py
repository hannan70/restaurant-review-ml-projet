from django.db import models

# Create your models here.
class Feedback(models.Model):
    review_text = models.CharField(max_length=200)
    sentiment = models.CharField(max_length=20)

    def __str__(self):
        return self.review_text

    
