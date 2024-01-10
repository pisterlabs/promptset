import datetime
import os
import openai

from flask import abort, current_app, jsonify, request
from flask_restful import Resource, fields, marshal_with
from moodapp.logic.models import db, Mood
from .parsers import (
    mood_get_parser,
    mood_post_parser,
)


mood_fields = {
    'id': fields.Integer(),
    'date': fields.DateTime(dt_format='iso8601'),
    'question': fields.String(),
    'answer': fields.String(),
    'output': fields.String(),
    'value': fields.Integer(),
}

basedir = os.path.abspath(os.path.dirname(__file__))
openai.api_key = os.environ.get('OPENAI_KEY')


class MoodApi(Resource):
    @marshal_with(mood_fields)
    def get(self):
        args = mood_get_parser.parse_args()

        moods = Mood.query.all()

        if not moods:
            abort(404)

        return moods

    def post(self):
        args = mood_post_parser.parse_args(strict=True)
        if not args:
            abort(404)

        question = args['question']
        answer = args['answer']

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"Computer asks user a question, then user answers the question. \nAnswer is mapped to numerical value between 1 and 5.\n1 means bad, 5 means good.\nWarm feedback is given based on the answer.\n\nComputer: How are you feeling today?\nUser: I'm feeling good.\nNumerical: 4\nFeedback: That's good to hear!\n\nComputer: How did you sleep?\nUser: Not that well.\nNumerical: 2\nFeedback: I'm sorry to hear that. \n\nComputer: {question}\nUser: {answer}\n",
            temperature=0.7,
            max_tokens=22,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        output = response.choices[0].text.split("\n")
        numerical = int(output[0].split(": ")[-1])
        feedback = output[1].split(": ")[-1]

        new_mood = Mood(question, answer, feedback, numerical)

        db.session.add(new_mood)
        db.session.commit()
        return {'feedback': feedback}, 201
