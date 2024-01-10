"""
This module takes care of starting the API Server, Loading the DB and Adding the endpoints
"""
from flask import Flask, request, jsonify, url_for, Blueprint, session
from api.models import db, User, Post, Favorites, Comment
from flask_cors import CORS
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
import sys
import openai
import os
from .models import Image
from config import API_KEY

api = Blueprint('api', __name__)
app = Flask(__name__)
openai.api_key = (API_KEY) #stored securely in config.py which is in the gitignore list
openai.Model.list()


@api.route('/signup', methods=['POST'])
def signup():
    # Retrieve request data
    username = request.json.get('username')
    password = request.json.get('password')

    # Check if the email is already registered
    if User.query.filter_by(username=username).first():
        return jsonify(message='Username already registered'), 409  

    # Create a new user object
    new_user = User(username=username, password=password)

    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(sys.exc_info())
        return jsonify(message='Failed to register user'), 500

    user_id = new_user.id

    return jsonify(message='User registered successfully', user_id=user_id), 201

@api.route('/login', methods=['POST'])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    # Perform authentication
    user = User.query.filter_by(username=username).first()

    if user is None or not password == user.password:
        if user is None or not user.check_password(password):
            return jsonify({"msg": "Incorrect email or password"}), 401

    # Generate access token
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, user_id=user.id)


@api.route('/generate_image', methods=['POST'])
#@jwt_required
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')
    number = data.get('number', 1)
    size = data.get('size', '512x512')
    response_format = data.get('response_format', 'url')  # Change response_format to 'url'
    
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=number,
            size=size,
            response_format=response_format
        )

        urls = []
        if response_format == "url":
            urls = [data['url'] for data in response.data]

        response_headers = {
            'Access-Control-Allow-Methods': 'POST'
        }

        return jsonify(urls), 200, response_headers
    except Exception as e:
        return jsonify(error=str(e)), 500

@api.route('/comments', methods=['POST'])
def comments():
    data = request.get_json()
    print("I AM DATA COMMENTS",data)
    text = data.get('text')
    user_id = data.get('user_id')
    post_id = data.get('post_id')

    new_comment = Comment(
        text=text,
        user_id=user_id,
        post_id=post_id,
    )

    db.session.add(new_comment)
    db.session.commit()

    return jsonify(new_comment.serialize()), 200


@api.route('/comments', methods=['GET'])
def get_comments():
    comment_list= Comment.query.all()
    all_comments= list(map(lambda comment:comment.serialize(),comment_list))
    return jsonify(all_comments), 200

@api.route('/users/<int:id>', methods=['GET'])
@jwt_required()
def get_user_(id):
    user = User.query.filter_by(id=id).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    request_token = request.headers.get('Authorization', '').split('Bearer ')[1] if 'Authorization' in request.headers else None
    if 'access_token' not in session or session['access_token'] != request_token:
        if request_token:
            session['access_token'] = request_token
        return jsonify({'message': 'Invalid access token'}), 401

    return jsonify(user.serialize()), 200

@api.route('/posts', methods=['GET'])
def get_posts():
    posts = Post.query.all()
    serialized_posts=[]

    for post in posts: 
        serialized_posts.append(posts.serialize())
    return jsonify(serialized_posts), 200

@api.route('/posts', methods=['POST'])
@jwt_required()
def create_post():
    data = request.get_json()
    user_id = data.get('user_id')
    post_id = data.get('post_id')

    user = User.query.filter_by(id=user_id).first()


    if not user:
        return jsonify({'message': 'User not found'}), 404


    post = Post(
        title=data['title'],
        content=data['content'],
        author=user,
        post_id=post_id
    )

    db.session.add(post)
    db.session.commit()

    return jsonify({'message': 'Post created successfully', 'post_id': post.id}), 200



# @api.route("/post_images", methods=["POST"])
# def create_post_image():
#     image = request.files['file']
#     post_id = request.form.get("post_id")
#     response = uploader.upload(
#         image,
#         resource_type="image",
#         folder="posts"
#     )
#     new_post_image = Image(
#         post_id=post_id,
#         url=response["secure_url"],
#     )
#     db.session.add(new_post_image)
#     db.session.commit()

#     return jsonify(new_post_image.serialize()), 201


@api.route('/single/<int:theid>', methods=['GET'])
def get_single(theid):
    item = User.query.get(theid)
    if not item:
        return jsonify({'message': 'Item not found'}), 404


    return jsonify({'item': item.serialize()}), 200

@api.route('/users/favorites', methods=['POST'])
@jwt_required()
def add_favorite():
    data = request.get_json()
    print(data)

    user_id = data.get('user_id')
    print (user_id)
    # Check if user_id is provided and is an integer
    if not user_id or not isinstance(user_id, int):
        return jsonify({'message': 'Invalid user ID'}), 400

    user = User.query.filter_by(id=user_id).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    post_id = data.get('post_id')

    if not post_id or not isinstance(post_id, int):
        
        return jsonify({'message': 'Invalid post ID'}), 400
        
    favorite = Favorites(
        user_id=user_id,
        post_id=post_id,
    )
    
    db.session.add(favorite)
    db.session.commit()

    favorites = Favorites.query.filter_by(user_id=user_id).all()  # Use .all() to get all favorites

    # Serialize the list of favorites
    favorites_data = [favorite.serialize() for favorite in favorites]

    return jsonify({'message': 'Favorite added successfully', 'favorites': favorites_data}), 200

@api.route('/users/favorites/<int:id>', methods=['DELETE'])
@jwt_required()
def delete_favorite(id):
    current_user_id = get_jwt_identity()
    favorite = Favorites.query.get(id)

    if not favorite:
        return jsonify({'message': 'Favorite not found'}), 404

    db.session.delete(favorite)
    db.session.commit()

    return jsonify({'message': 'Favorite deleted successfully'}), 200

@api.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    # Remove the stored access token from the session
    session.pop('access_token', None)
    return jsonify({'message': 'Logged out successfully'}), 200


@api.route('/hello', methods=['GET'])
@jwt_required()
def hello():
    # Retrieve the username from the token
    username = get_jwt_identity()

    # Create the message with the username
    message = f"Hello, {username}"

    # Return the message as JSON response
    return jsonify({'message': message}), 200

if __name__ == "__main__":
    api.run()

