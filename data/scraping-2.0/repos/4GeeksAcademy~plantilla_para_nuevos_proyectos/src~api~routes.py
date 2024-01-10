"""
This module takes care of starting the API Server, Loading the DB and Adding the endpoints
"""
import os
from flask import Flask, request, jsonify, url_for, Blueprint, current_app
from api.models import db, User
from api.images import Imagen
from api.favoritos import Favoritos
from api.utils import generate_sitemap, APIException

from api.extensions import jwt, bcrypt
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager

import cloudinary
import cloudinary.uploader
import cloudinary.api

import openai 

cloudinary.config(
cloud_name = os.getenv("CLOUDINARY_NAME"),
api_key = os.getenv("CLOUDINARY_KEY"),
api_secret = os.getenv("CLOUDINARY_SECRET"),
api_proxy = "http://proxy.server:9999"
)

openai.api_key = os.environ.get('OPENAI_API_KEY')

api = Blueprint('api', __name__)


@api.route('/hello', methods=['POST', 'GET'])
def handle_hello():
    password_encrypted = bcrypt.generate_password_hash("123",10).decode("utf-8")
    response_body = {
        "message": password_encrypted
    }

    return jsonify(response_body), 200

@api.route('/hola', methods=['POST', 'GET'])
def handle_hola():

    response_body = {
        "message": "Hello! I'm a message that came from the backend, check the network tab on the google inspector and you will see the GET request"
    }

    return jsonify(response_body), 200

@api.route('/login', methods=['POST'])
def login():
    body = request.get_json()
    email = body['email']
    password = body['password']

    user = User.query.filter_by(email=email).first()

    if user is None:
        raise APIException("usuario no existe", status_code=401)
    
    #validamos el password si el usuario existe y si coincide con el de la BD
    if not bcrypt.check_password_hash(user.password, password):
        raise APIException("usuario o password no coinciden", status_code=401)

    access_token = create_access_token(identity= user.id)
    return jsonify({"token": access_token, "email":user.email}), 200

@api.route('/signup' , methods=['POST'])
def signup():
    body = request.get_json()
    print(body)
    #print(body['username'])     
    try:
        if body is None:
            raise APIException("Body está vacío o email no viene en el body, es inválido" , status_code=400)
        if body['email'] is None or body['email']=="":
            raise APIException("email es inválido" , status_code=400)
        if body['password'] is None or body['password']=="":
            raise APIException("password es inválido" , status_code=400)      
      

        password = bcrypt.generate_password_hash(body['password'], 10).decode("utf-8")
        
        new_user = User(email=body['email'], password=password, is_active=True)
       

        user = User.query.filter_by(email=body['email'])
        if not user:
            raise APIException("El usuario ya existe" , status_code=400)    

        print(new_user)
        #print(new_user.serialize())
        db.session.add(new_user) 
        db.session.commit()
        return jsonify({"mensaje": "Usuario creado exitosamente"}), 201

    except Exception as err:
        db.session.rollback()
        print(err)
        return jsonify({"mensaje": "error al registrar usuario"}), 500

@api.route('/upload', methods=['POST'])
def handle_upload():

    if 'image' not in request.files:
        raise APIException("No image to upload")

    print("FORMA DEL ARCHIVO: \n",  request.files['image'])
    my_image = Imagen()

    result = cloudinary.uploader.upload(
        request.files['image'],
        public_id=f'sample_folder/profile/my-image-name',
        crop='limit',
        width=450,
        height=450,
        eager=[{
            'width': 200, 'height': 200,
            'crop': 'thumb', 'gravity': 'face',
            'radius': 100
        },
        ],
        tags=['profile_picture']
    )

    my_image.ruta = result['secure_url']
    my_image.user_id = 1 # Aquí debería extraer del token, el id del usuario
    db.session.add(my_image) 
    db.session.commit()

    return jsonify(my_image.serialize()), 200

@api.route('/image-list', methods=['GET'])
def handle_image_list():
    images = Imagen.query.all() #Objeto de SQLAlchemy
    images = list(map(lambda item: item.serialize(), images))

    response_body={
        "lista": images
    }
    return jsonify(response_body), 200

@api.route('/chatgpt', methods=['POST'])
def open_ai():
    body =request.get_json()    
    prompt = "Eres una página web de citas para cuidar mascotas, responde acorde a esto, según lo que te pregunte el usuario: "+ body['prompt']

    completation = openai.Completion.create(engine="text-davinci-003",
                            prompt=prompt,
                            n=1,
                            max_tokens=2048)
    
    #print(completation.choices[0])
    print(completation.choices[0].text)
    response = {
        "message":completation.choices[0].text
    }
    return jsonify(response), 200

# MPT-7b : 64k tokens, ggml, q4_0, 128bits 4Q 
# Oobaboonga, Koboldcpp
