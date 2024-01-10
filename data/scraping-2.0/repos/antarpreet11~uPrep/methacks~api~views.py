from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .models import User
import cohere 
import json
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
# from annoy import AnnoyIndex
import warnings
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate
import cohere

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

load_dotenv()
api_key = os.environ.get("COHERE_API_KEY")
model_id = os.environ.get("COHERE_MODEL_ID")
co = cohere.Client(api_key)

headers = {"Authorization": f"Bearer {api_key}"}
User = get_user_model()
# Create your views here.

# view list of all diff rooms

def main(request):
    def preprocess(text):
        # convert to lowercase
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # remove stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return " ".join(filtered_text)
    return HttpResponse("Hello")

@csrf_exempt
def getClassification(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text')
        response = co.classify(
            model=model_id,
            inputs=[text])
        
        prediction = response.classifications[0].prediction
        confidence = response.classifications[0].confidence

        score = getScore(prediction, confidence)

        return JsonResponse({'score': score})
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def getScore(pred, conf): 
    confidence = 0
    if pred == ' right':
        confidence = min((conf * 140), 90)
    else:
        confidence = max((1 - conf) * 50, 20)
    
    return confidence

@csrf_exempt
def signup_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already exists'}, status=400)

        user = User.objects.create_user(username=username, email=email, password=password)

        return JsonResponse({'message': 'User created successfully'})

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Authentication successful
            return JsonResponse({'message': 'Login successful'})
        else:
            # Authentication failed
            return JsonResponse({'error': 'Invalid credentials'}, status=401)

    return JsonResponse({'error': 'Invalid request method'}, status=405)