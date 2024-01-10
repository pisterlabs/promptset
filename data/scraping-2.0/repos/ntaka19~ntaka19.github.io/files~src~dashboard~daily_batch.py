from openai import OpenAI
import json
import os
import sys
import requests
import django
from django.conf import settings
from django.template import Template, Context
from datetime import datetime
from pytz import timezone


class ChatGPTWrapper:
    
    def __init__(self):
        self.api = os.environ['OPENAI_API']

    def GetResponse(self, prompt):
        client = OpenAI(
            api_key = self.api
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )

        return chat_completion.choices[0].message.content
        #print(chat_completion["choices"][0]["message"]["content"] )

        """
        openai.api_key = self.api

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt
                }, #※1後述
            ]
        )
        """
        #return response["choices"][0]["message"]["content"] #返信のみを出力



def forecast():
    url = 'https://api.open-meteo.com/v1/forecast?latitude=35.69&longitude=139.69&hourly=temperature_2m,rain,showers,weathercode&forecast_days=1&timezone=Asia%2FTokyo'
    response = requests.get(url)
    data = json.loads(response.text)
    
    today = datetime.fromisoformat(data['hourly']['time'][0]).strftime('%m-%d %a')
    updated_time = datetime.now(timezone("Asia/Tokyo")).strftime('%m/%d %H:%M')

    chatgpt = ChatGPTWrapper()
    prompt = "この日の天気を簡潔にキャスターのように予報をして：  {first}".format(first=json.dumps(data))                                
    forecast_text = chatgpt.GetResponse(prompt)

    ##html生成 あとで別のモジュールにしておく。
    settings.configure(
            DEBUG=True,
            TEMPLATES=[
                {
                    'BACKEND': 'django.template.backends.django.DjangoTemplates',
                    'APP_DIRS': True,
                },
            ]
        )
    django.setup()

    with open('./files/src/dashboard/template_todayweather.html', 'r') as template_file:
        t = Template(template_file.read())
        c = Context({"today" : today,
                     "forecast_text": forecast_text,
                     "updated_time": updated_time})    
        
        rendered_html = t.render(c)

        # Save the rendered HTML as an HTML file
        with open("./docs/src/dashboard/forecast_text.html", "w") as file:
            file.write(rendered_html)


def main():
    forecast()


if __name__ == "__main__":
    main()
