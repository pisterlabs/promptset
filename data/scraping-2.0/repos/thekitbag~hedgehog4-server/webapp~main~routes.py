
from webapp import db
from webapp.models import Result
from webapp.main import bp

from webapp.main.mocks.places import results as google_places
from webapp.main.mocks.place_details import details as google_place_details
from webapp.main.response_handlers.google_places import get_fields

import time
import openai
import os

from flask import json, request, current_app

@bp.route('/health')
def health():
    return '', 200

@bp.route('/ready')
def ready():
    return '', 200

@bp.route('/test', methods=['GET'])
def test():
    return 'Working'

@bp.route('/search', methods=['GET', 'POST'])
def search():
    r = json.loads(request.data.decode('utf-8'))
    search_term = r['search_term']

    places = get_fields(google_places)

    results = {'search_term': search_term, 'places': places}
        
    return results

@bp.route('/place_details', methods=['GET'])
def place_details():
    return {'details': google_place_details['result']}

@bp.route('/place_info_request', methods=['GET'])
def place_info_request():
    openai.api_key = current_app.config['GPT_API_KEY']

    system_message = "You are placeInfoGPT, an AI assistant that looks at various sources including online reviews and \
                        answers questions about bars, hotels and restaurants to help people decide if they should go there or not. \
                        If you cannot find the answer just how the user can find the answer for themselves, giving helful information such \
                        as website URLs and phone numbers"
    user_message = request.args.get("q")
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ]
                    )
    print(response)
    return response["choices"][0]["message"]["content"]
