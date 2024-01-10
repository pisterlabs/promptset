import openai
import pandas as pd
from flask import Flask, request

api_key = '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'

'''
to run the script, run this:
flask --app categorize run
to access the endpoint, run this:
url: {endpoint}/predict?key={api_key}&content={string of tweet}
OR
python:
import requests
url = f"{endpoint}/predict?key={api_key}&content={string of tweet}"
requests.get(url)
'''

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def getSinglePrediction() -> str:
  openai.api_key = request.args.get('key')
  content = request.args.get('content')

  lstOfCategories = ['web3 activities and events', 'web3 announcements', 'web3 research output', 'web3 meme', 'crypto and markets', 'web3 phishing or irrelevant', 'unknown']

  response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Decide which category the Tweet best classify as {lstOfCategories}.\n\nTweet: \"{content}\"\nCategory: ",
      temperature=0,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0.5,
      presence_penalty=0
    )

  try:
    result = response['choices'][0]['text']
    return result[-len(result)+1:]
  except:
     return f"Error: {response}"

def getPredictionDF(apiKey, csv_path, return_path = 'categorized.csv'):
  openai.api_key = apiKey

  lstOfCategories = ['web3 activities and events', 'web3 announcements', 'web3 research output', 'web3 meme', 'crypto and markets', 'web3 phishing or irrelevant', 'unknown']

  currDF = pd.read_csv(csv_path)
  returnDF = pd.DataFrame(columns=['user', 'tweet_content', 'category'])

  for _, row in currDF.iterrows():
    content = row['tweet_content']

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Decide which category the Tweet best classify as {lstOfCategories}.\n\nTweet: \"{content}\"\nCategory: ",
      temperature=0,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0.5,
      presence_penalty=0
    )

    result = response['choices'][0]['text']

    returnDF.loc[len(returnDF)] = {'user': row['user'], 'tweet_content': row['tweet_content'], 'category': result[-len(result)+1:]}

  returnDF.to_csv(return_path, index=False)