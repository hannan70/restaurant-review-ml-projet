from django import forms
from .models import Feedback


class FeedBackForm(forms.ModelForm):
    model = Feedback
    fields = "__all_"