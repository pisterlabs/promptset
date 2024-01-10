import os
import openai
import json
from application import db, app
from application.models import *
from application.utils import *
from flask import jsonify, make_response, request, render_template, redirect, session
from flask_restful import Resource
from nltk.probability import FreqDist
from werkzeug.exceptions import HTTPException

openai.api_key = os.environ['OPENAI_API_KEY']


class UserAPI(Resource):

  def get(self, email=None):
    if email != None:
      if session['email'] == email or (session['email'] != None and session['status'] in [1]):
        query = User.query.filter_by(email=email).first()
        return user_schema.jsonify(query)
      else:
        return make_response({}, 401)

  def post(self):
    if User.query.filter_by(email=request.json['email']).first() == None:
      query = User(email=request.json['email'],
                   password=request.json['password'])
      db.session.add(query)
      db.session.commit()
      return user_schema.jsonify(query)
    else:
      return make_response({}, 400)


class TranscriptAPI(Resource):

  def get(self, id=None):
    if session['email'] == None:
      return make_response({}, 401)
    else:
      if id == None:
        query = Transcript.query.all()
        serialized_query = transcripts_schema.dump(query)
        return jsonify(sorted(serialized_query, key=lambda x: x['id'], reverse=True))
      else:
        query = Transcript.query.filter_by(id=id).first()
        return transcript_schema.jsonify(query)

  def post(self):
    if session['email'] == None:
      return make_response({}, 401)
    else:
      file = request.files['file']
      text = openai.audio.translations.create(
        model="whisper-1",
        file=(file.filename,file.read(),file.content_type)
      )
      query = Transcript(user_email=session['email'],
                     text=text.text)
      db.session.add(query)
      db.session.commit()
      return transcript_schema.jsonify(query)


class AnalysisAPI(Resource):

  def get(self):
    if session['email'] == None:
      return make_response({}, 401)
    else:
      query = Transcript.query.all()
      serialized_query = transcripts_schema.dump(query)
      data = {
        'user' : [],
        'other' : []
      }
      for transcript in serialized_query:
        if transcript['user_email'] == session['email']:
          data['user'] += preprocess(transcript['text'])
        else:
          if transcript['user_email'] not in data.keys():
            data[transcript['user_email']] = []
          data[transcript['user_email']] += preprocess(transcript['text'])
          data['other'] += preprocess(transcript['text'])
      phrases = extract_phrases(' '.join(data['user']))
      analysis = {
        'freq_user' : FreqDist(data['user']).most_common(10),
        'freq_other' : FreqDist(data['other']).most_common(10),
        'phrases' : [' '.join(phrase[0]) for phrase in FreqDist(phrases).most_common(10)],
        'similarity' : similarity(data),
      }
      return jsonify(analysis)


class HomeAPI(Resource):

  def get(self):
    return make_response(render_template('index.html'), 200)

  def post(self):
    if 'role' in request.json:
      response = UserAPI().post()
      if response.status_code == 400:
        return make_response({}, 400)
      elif response.status_code == 200:
        session['email'] = response.json['email']
        return make_response({}, 200)
    else:
      response = User.query.filter_by(email=request.json['email']).first()
      if response == None:
        return make_response({}, 404)
      else:
        if response.password == request.json['password']:
          session['email'] = response.email
          return make_response({}, 200)
        else:
          return make_response({}, 400)


@app.route('/logout')
def logout():
  session['email'] = None
  return redirect('/')


@app.route('/current')
def current():
  c = {
    'email': session['email'],
  }
  return json.dumps(c)


@app.errorhandler(HTTPException)
def handle_exception(e):
  response = e.get_response()
  response.data = json.dumps({
    "code": e.code,
    "name": e.name,
    "description": e.description,
  })
  response.content_type = "application/json"
  return make_response(response)
