from flask import Blueprint, render_template, jsonify, request
from dotenv import load_dotenv
from app.services.weather import Weather
from app.services.openai import Openai

main = Blueprint('main', __name__)

load_dotenv()

@main.route('/')
def index():
  return render_template('main.html')

@main.route('/rec', methods = ['POST', 'GET'])
def get_rec():
  try:
    city = request.get_json()['city']
    weather = Weather()
    resp = weather.get_weather(city)
    if resp.status_code == 200:
      weather = resp.json()
      openai = Openai()
      rec = openai.get_chat_rec(weather)
      return jsonify(rec.choices[0].message.content)
    
    else:
      print(f'Danger Will Robinson: {resp.status_code}')
      return 'Oops, there was an error!', resp.status_code
  except Exception as e:
    print('Oops, there was an error with external api calls!', e)
    return 'Oops, there was an error with external api calls!', 500 
