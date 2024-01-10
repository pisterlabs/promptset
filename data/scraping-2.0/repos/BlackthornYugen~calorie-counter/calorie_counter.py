#!/usr/bin/env python3

import os
import openai

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import NoResultFound

from app.models import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SQLALCHEMY_DATABASE_URI", default='sqlite:///:memory:')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get("FLASK_COOKIE_SECRET", default="this-default-will-let-anyone-forge-a-cookie")
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24  # 24 hours

openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

db = SQLAlchemy(app, model_class=User)
default_gpt_model = os.environ.get("OPENAI_API_MODEL", default="gpt-4")


@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve logs."""
    logs = db.session.query(FoodLog).all()
    return jsonify([log.to_dict() for log in logs])  # Assuming you have a method to_dict in FoodLog model


@app.route('/logs', methods=['POST'])
def create_log():
    """Endpoint to create a new log."""
    data = request.json
    new_log = FoodLog(
        user_id=data['user_id'],
        food_id=data['food_id'],
        quantity_in_grams=data['quantity_in_grams'],
        log_time=datetime.strptime(data['log_time'], '%Y-%m-%dT%H:%M:%S'),
        meal_type=data['meal_type']
    )
    db.session.add(new_log)
    db.session.commit()
    return jsonify({"message": "Log created successfully", "log_id": new_log.log_id}), 201


@app.route('/users', methods=['GET'])
def get_users():
    """Endpoint to retrieve users."""
    users = db.session.query(User).all()
    return jsonify([user.to_dict() for user in users])


@app.route('/foods', methods=['GET'])
def get_foods():
    """Endpoint to retrieve foods."""
    items = db.session.query(Food).all()
    return jsonify([item.to_dict() for item in items])


@app.route('/gpt', methods=['POST'])
def call_gpt():
    request_query = str(request.json["message"])
    request_model = request.json.get("model", default_gpt_model)

    try:
        # Check if the query and model are in the cache and if it's recent
        cache_entry = db.session.query(GPTCache).filter(
            GPTCache.query == request_query,
            GPTCache.model == request_model
        ).one()

        if cache_entry.is_recent():
            return cache_entry.response  # Use the cached response
    except NoResultFound:
        # If not in cache or not recent, proceed to make a new GPT request
        food = get_foods()
        messages = [{
            "role": "system",
            "content": """
                Create a json object with nutritional information about what is described to you. Sometimes
                you will need to guess the nutritional information. For liquids prefer 100 grams. For example, 
                if a user wants to track a glass of milk, and there isn't the food item in the context. New 
                foods should also be updated when a new conversion is needed, but in that case you can skip
                macros and calories if they are not changed. Example:

                "A glass of milk and a cookie"

                { "new_foods": [ {
                    "name": "1% Milk",
                    "quantity": 100,
                    "quantity_units": "grams",
                    "calories": 123.33333333333334,
                    "conversions": { "cups": 0.425 },
                    "macros": {
                        "carbs": 0.002,
                        "fat": 10.7,
                        "protein": 0.02
                    }, {
                    "name": "Chocolate Chip Cookie",
                    "quantity": 1,
                    "quantity_units": "cookies",
                    "calories": 4000,
                    "conversions": { "grams": 50 },
                    "macros": {
                        "carbs": 30,
                        "fat": 20,
                        "protein": 3
                    }
                ], "journal_entry": [{
                        "food": "milk", 
                        "quantity": 2,
                        "quantity_units": "cups",
                        "comments": ["Assuming 1% milk", "Assuming a glass is 2 cups"] 
                    },
                    {
                        "food": "cookie", 
                        "quantity": 1,
                        "quantity_units": "cookies",
                        "comments": ["Assuming a 50g chocolate chip cookie"] 
                    }],
                  "warnings": []
                }

                If all food that will be journaled is provided, just give an empty array of new foods. Warnings
                can be given as an array of strings when there is a problem. Try to keep the response valid json.
                """
        }, {
            "role": "user",
            "content": " Here are the foods I know of: \n" + str(food.json)
        }, {
            "role": "user",
            "content": "Here is what I want to journal: \n" + str(request.json["message"])
        }]

        chat_model = openai.Model.retrieve(request_model)
        completion = openai.ChatCompletion.create(
            model=chat_model.id,
            messages=messages
        )

        # Cache the new response
        new_cache_entry = GPTCache(
            query=request_query,
            model=request_model,
            response=completion
        )
        db.session.add(new_cache_entry)
        db.session.commit()

        return completion


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
