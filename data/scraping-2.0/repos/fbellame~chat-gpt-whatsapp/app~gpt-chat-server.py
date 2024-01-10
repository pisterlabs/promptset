from flask import Flask, request
from apiflask import HTTPError
import openai
import os
from twilio.twiml.messaging_response import MessagingResponse

OPENAI_API_KEY = "OPENAI_API_KEY"
USAGE_MESSAGE = "Welcome to the chat GPT app! Call POST /chat?token=[CHAT_TOKEN] with query in the body"
GPT_MODEL = "text-davinci-003"
MY_ORGANISATION = "MY_ORGANISATION"
CHAT_TOKEN = os.getenv("CHAT_TOKEN")
MAX_TOKEN = int(os.getenv("MAX_TOKEN", default=30))

app = Flask(__name__)
openai.api_key = os.getenv(OPENAI_API_KEY)
openai.organization = os.getenv(MY_ORGANISATION)

@app.route("/")
def hello():
  return USAGE_MESSAGE

@app.route("/chat", methods=['GET','POST'])
def chat():
      
  token = request.args.get('token')
  if valid_token(token):
      
    if request.method == 'POST':
      question = request.form['Body']
          
      response = openai.Completion.create(
        model=GPT_MODEL,
        prompt=question,
        max_tokens=MAX_TOKEN,
        temperature=0.6)
          
      print(str(response))
      resp = MessagingResponse()
      resp.message(response.choices[0].text)    
          
      return str(resp)
  else:
      raise HTTPError(403, message="Denied, invalid token!!!")
  
  return USAGE_MESSAGE

def valid_token(token):
      return token == CHAT_TOKEN

if __name__ == "__main__":
      app.run(host="0.0.0.0", port=80)
