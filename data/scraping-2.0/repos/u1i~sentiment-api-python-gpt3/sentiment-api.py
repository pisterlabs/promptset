import os
import openai
import json
from flask import Flask, request

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_sentiment(input_text):

	response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="analyzing sentiment from user input. The possible results are: positive,negative,neutral,upset\n\ntext:the product is terrible I want my money back!!!\nsentiment:upset\n\ntext:loved it, will always buy again thanks!\nsentiment:positive\n\ntext:everything ok thank you\nsentiment:neutral\n\ntext:wasn't very happy about the delay but overall ok\nsentiment:negative\n\ntext:" + input_text + "\nsentiment:",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

	return(response["choices"][0]["text"])

app = Flask(__name__)
@app.route('/sentiment', methods=['POST'])
def apirequest():

	if not request.json or not 'text' in request.json:
		print(request.json)
		abort(400)
	sentiment=analyze_sentiment(request.json['text'])
	return json.dumps({'sentiment': sentiment})
app.run(port=8080)
