from django.db import models
from django.utils import timezone
import datetime
from django.contrib import admin
import os
import openai
import requests
from django.contrib.auth import get_user_model
from webpush import send_user_notification, send_group_notification
from webpush.utils import send_to_subscription
import json
with open('./.env') as f:
    for line in f:
        key, value = line.strip().split('=')
        os.environ[key] = value
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published")
    @admin.display(
        boolean=True,
        ordering="pub_date",
        description="Published recently?",
    )
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    
class Querys(models.Model):
    studentID = models.CharField(max_length=50)
    name = models.CharField(max_length=200)
    query = models.CharField(max_length=200)
    


def answer(query):
  openai.api_key = os.getenv("OPENAI_API_KEY")
  response = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      messages = [
          {"role": "system", "content": "Please answer the questions below in one line.\n"},
          {"role": "user", "content": query}
      ]
  )
  return response['choices'][0]['message']['content']



def get_translate(text, language):
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")
    if language: a='ko'; b = "en"
    else: a='en'; b = "ko"
    data = {'text' : text,#inputtext
            'source' : a,#input lan
            'target': b}#output lan

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data= data)
    rescode = response.status_code

    if(rescode==200):
        t_data = response.json()
        return response.json()['message']['result']['translatedText']
    else:
        print("Error Code:" , rescode)
        return 0
    


def episode_webpush(episode):

    User = get_user_model()
    users = User.objects.all()

    payload = {"head": f"{episode.__str__()}이 업데이트 되었습니다.", "body": f"{episode.__str__()} 업데이트 일시 : {episode.modify_date}", 
            "icon": "https://i.imgur.com/...", "url": f"{episode.get_absolute_url()}"}
     
    payload = json.dumps(payload) # json으로 변환 https://github.com/safwanrahman/django-webpush/issues/71

    for user in users:
        push_infos = user.webpush_info.select_related("subscription") 

        for push_info in push_infos:
            send_to_subscription(push_info.subscription, payload)