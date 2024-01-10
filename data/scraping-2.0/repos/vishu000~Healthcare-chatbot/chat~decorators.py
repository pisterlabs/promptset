from functools import wraps
import openai
from django.shortcuts import render, redirect
from django.contrib import messages
from openai.error import AuthenticationError, RateLimitError

def validate_api_key(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        api_key = request.session.get('api_key')
        openai.api_key = api_key

        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ]
            )
            return view_func(request, *args, **kwargs)
        except openai.error.AuthenticationError:
            messages.info(request, 'Invalid API key')
            return redirect('/')
        except openai.error.RateLimitError:
            messages.info(request, 'API key has expired')
            return redirect('/') 
    return wrapper
