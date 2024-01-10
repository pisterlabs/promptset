
import os
from flask import Blueprint, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, create_access_token
from api.api import create_api
from api.services.factory import create_user_service, create_recommendation_service
from api.services.user_service import InvalidCredentialsError
from sqlalchemy.exc import OperationalError
from data.models import db
import psycopg2
import boto3
import uuid
import openai

bp = Blueprint('bp', __name__)

user_service = create_user_service(db)
recommendation_service = create_recommendation_service(db)

        
# TODO get restaurants
# TODO get user's meals
# TODO get recommendations

@bp.route('/recommend_test', methods=['POST'])
def recommend_meal_test():
    user_id = 1
    response = recommendation_service.get_recommendation(request.json, user_id)

    return jsonify(response)


@bp.route('/meal_test', methods=['POST'])
def log_meal_test():
    user_id = 1
    print("logging test meal")
    print(f"req: {request.form}")
    try:
        response = recommendation_service.add_meal(request, user_id)
        return jsonify(response)
    except Exception as e:
        print(e)

@bp.route('/recommend', methods=['POST'])
@jwt_required()
def recommend_meal():
    user_id = get_jwt_identity()
    print(f"recommending meal for user {user_id}")
    response = recommendation_service.get_recommendation(request.json, user_id)

    return jsonify(response)

@bp.route('/meal', methods=['DELETE'])
@jwt_required()
def delete_meal():
    user_id = get_jwt_identity()
    print(f"logging meal for user {user_id}")
    try:
        response = recommendation_service.delete_meal(request, user_id)
        return jsonify(response)
    except Exception as e:
        print(e)


@bp.route('/meal', methods=['POST'])
@jwt_required()
def log_meal():
    user_id = get_jwt_identity()
    print(f"logging meal for user {user_id}")
    try:
        response = recommendation_service.add_meal(request, user_id)
        return jsonify(response)
    except Exception as e:
        print(e)

@bp.route('/meals', methods=['GET'])
@jwt_required()
def get_meals():
    user_id = get_jwt_identity()
    page_size = request.json.get('page-size', 10)
    page_number = request.json.get('page-number', 1)
    try:
        response = recommendation_service.get_meals(user_id, page_number, page_size)
        return jsonify(response)
    except Exception as e:
        print(e)

@bp.route('/restaurants', methods=['GET'])
@jwt_required()
def get_restaurants():
    user_id = get_jwt_identity()
    page_size = request.json.get('page-size')
    page_number = request.json.get('page-number')
    try:
        response = recommendation_service.get_restaurants(user_id, page_number, page_size)
        return jsonify(response)
    except Exception as e:
        print(e)

@bp.route('/restaurant', methods=['POST'])
@jwt_required()
def add_restaurant():
    user_id = get_jwt_identity()
    try:
        response = recommendation_service.add_restaurant(request)
        return jsonify(response)
    except Exception as e:
        print(e)

@bp.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    email = request.json.get('email')
    phone = request.json.get('phone-number')
    password = request.json.get('password')

    try:
        login_result = user_service.login(username, email, phone, password)
        return jsonify(login_result), 200
    except InvalidCredentialsError as e:
        return jsonify({"error": e.message}), 401

@bp.route('/register', methods=['POST'])
def register():
    data = request.json
    user = user_service.register(data['username'], data['email'], data['phone-number'], data['password'], data['name'])

    # Create JWT token
    access_token = create_access_token(identity=user.id)

    return jsonify(access_token=access_token), 201

def init_api_v1(app):
    app.register_blueprint(bp, url_prefix='/api/v1')
