from flask import render_template, request, jsonify
from app import app, db
from app.models import Conversation
import openai
from textblob import TextBlob

openai.api_key = app.config['OPENAI_API_KEY']

def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        response = generate_response(prompt)
        sentiment = analyze_sentiment(response)
        conversation = Conversation(prompt=prompt, response=response, sentiment=sentiment)
        db.session.add(conversation)
        db.session.commit()
        return jsonify({'response': response, 'sentiment': sentiment})
    return render_template('index.html')