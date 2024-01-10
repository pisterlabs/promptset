from twilio.rest import Client
import openai
import json


with open('app/main/resources/key.json', 'r') as f:
    keys = json.load(f)

account_sid = keys["twilio_sid"]
twilio_key = keys ["twilio_key"]
openai_key = keys ["openai_key"]
bot_number = keys ["phone_number"]

#Service configuring
client = Client(account_sid, twilio_key)
openai.api_key = openai_key
completion = openai.Completion()


def create_image(request):

    response = openai.Image.create(
        prompt=request,
        n=1,
        size="256x256",
    )

    return response["data"][0]["url"]

def answer(question):

    response = completion.create(
        prompt=question, model="text-davinci-003", temperature=0.9,
        top_p=1, frequency_penalty=0, presence_penalty=0.6, best_of=1,
        max_tokens=150)

    return response.choices[0].text.strip()









