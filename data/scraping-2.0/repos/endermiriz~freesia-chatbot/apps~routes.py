from apps import app
from flask import request
import openai

from twilio.twiml.messaging_response import MessagingResponse

openai.api_key = app.config.get("OPENAI_KEY")

def openai_create(prompt):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    return response.choices[0].text




@app.route("/msgpost",methods = ['POST'])
def get_message():
    userMsg = request.form.get('Body')
    userPhone = request.form.get('From', None)
    profileName = request.form.get('ProfileName', None)
    return gbt_reply(userMsg)

def gbt_reply(msg):
    output = openai_create(msg)
    resp = MessagingResponse()
    resp.message("Freesia: {}".format(output))
    return str(resp)