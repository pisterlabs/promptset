import os
import string
from django.http import HttpResponse
from django.utils import timezone
from django.conf import settings

# from celery import shared_task
from django.db.models import Q

import openai
import requests
session = requests.Session()
session.trust_env = False

from .models import Stories


# @shared_task
def create_stories(request: str, user=None):
    story: None
    is_exist, status: bool = False
    # Task to create stories
    try:
        story = Stories.objects.get(request=request)
        is_exist, status = True
    except Stories.DoesNotExist:
        request_story = chatgpt_create_stories(request=request)
        if request_story.status == True: 
            story = Stories.objects.create(
                user=user,
                request=request,
                content=request_story.content,
                ai_model=request_story.model,
                ai_role=request_story.role,
            )
            is_exist = False, 
            status = True,
        else:
            is_exist, status = False

    return {
        'status': status, 
        'is_exist': is_exist,
        'data': story,
    }


def chatgpt_create_stories(request: str):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    messages = [
        {"status": False, "role": "system", "content": "You are an intelligent assistance"}
    ]

    response = {"status": False, "role": "system", "content": ""}

    while True:
        try:
            user_query = request
            if user_query:
                messages.append(
                    {"role": "user", "content": user_query},
                )
                chat_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=messages
                )

            reply = chat_response.choices[0].message.content
            print(f"ChatGPT Response => {reply}")
            messages.append(
                {
                    "status": True,
                    "model": "gpt-3.5-turbo",
                    "role": "assistant", 
                    "content": reply
                }
            )
            response.status = True
            response.model = 'gpt-3.5-turbo'
            response.role = "assistant"
            response.content = reply

            print("Message Dict", messages)
            print("Reponse Object", response)
            return response
        except Exception as err:
            print("An error occurred with error message => ", err)
            exit('AI is stopping now...')
            return response