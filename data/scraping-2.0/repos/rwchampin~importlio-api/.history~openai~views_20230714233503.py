from django.shortcuts import render
import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY
# Create your views here.
