import os

import openai
from flask import Flask, redirect, render_template, request, url_for, jsonify
import requests

# app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

from flask import Flask, request, jsonify, url_for, Blueprint, current_app
from api.modelsChat import db, RecipeChat
from api.recipe import Recipe
# Recipe, Like, Coment, Favorito, RecipeIngredient
from api.user import User
from api.token_bloked_list import TokenBlokedList
from api.favoritos import Favorito
from api.utils import generate_sitemap, APIException

from api.extensions import jwt, bcrypt
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
import re

#PARA OPERACIONES CON FECHAS Y HORAS.
from datetime import date, time, datetime, timezone, timedelta #timedelta, es para hacer resta de horas.

#PARA MANEJAR LA ENCRIPTACIÓN DE LA INFORMACIÓN. ADICIONAL SE REQUIERE, FLASK, REQUEST, JSONIFY, SIN EMBARGO ESOS YA FUERON INSTALADOS ARRIBA.
from flask_jwt_extended import get_jwt
from flask_jwt_extended import JWTManager
# from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
import cloudinary.api

rrecipe = Blueprint('rrecipe', __name__)

# Configurar cloudinary
cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  api_proxy = "http://proxy.server:9999",
  secure = True
)


# Handle/serialize errors like a JSON object
@rrecipe.errorhandler(APIException)
def handle_invalid_usage(error):
    return jsonify(error.to_dict()), error.status_code


#Funcion de verificación de token:
def verificacionToken(identity):
    jti = identity["jti"]
    token = TokenBlokedList.query.filter_by(token=jti, is_blocked=True).first()
    
    if token:
        return True  # Token bloqueado
    else:
        return False  # Token no bloqueado

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@rrecipe.route('/AddRecipe', methods=['POST'])
@jwt_required()
def add_recipe():

    jwt_claims = get_jwt()
    print(jwt_claims)
    user = jwt_claims["users_id"]
    print("el id del USUARIO:",user)

    if 'image_of_recipe' not in request.files:
        raise APIException("No image to upload")
    if 'description' not in request.form:
        raise APIException("No description to upload")
    if 'name' not in request.form:
        raise APIException("No user_query to upload")
    
    # Consigue un timestamp y formatea como string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_cloudinary_url = cloudinary.uploader.upload(
        request.files['image_of_recipe'],
        public_id = f'{request.form.get("name").replace(" ", "_")}_{timestamp}',
    )['url']  # Extract the 'url' from the returned dictionary

    new_recipe_chat = RecipeChat(
        name=request.form.get("name"),  # actualiza esto
        description=request.form.get("description"),
        user_query=request.form.get("name"),
        image_of_recipe=image_cloudinary_url,  # now this is the URL of the image in Cloudinary
        share=False,
        generated_by_ia=False,
        user_id=user,
    )

    # Añadir y hacer commit a la nueva entrada
    db.session.add(new_recipe_chat)
    db.session.commit()

    # Retornar la receta, la URL de la imagen y el ID de la receta en la respuesta
    return jsonify({"recipe": request.form.get("user_query"), "image_url": image_cloudinary_url, "recipe_id": new_recipe_chat.id})

@rrecipe.route('/AddAndShareRecipe', methods=['POST'])
@jwt_required()
def add_and_share_recipe():

    jwt_claims = get_jwt()
    print(jwt_claims)
    user = jwt_claims["users_id"]
    print("el id del USUARIO:",user)

    if 'image_of_recipe' not in request.files:
        raise APIException("No image to upload")
    if 'description' not in request.form:
        raise APIException("No description to upload")
    if 'name' not in request.form:
        raise APIException("No user_query to upload")
    
    # Consigue un timestamp y formatea como string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_cloudinary_url = cloudinary.uploader.upload(
        request.files['image_of_recipe'],
        public_id = f'{request.form.get("name").replace(" ", "_")}_{timestamp}',
    )['url']  # Extract the 'url' from the returned dictionary

    new_recipe_chat = RecipeChat(
        name=request.form.get("name"),  # actualiza esto
        description=request.form.get("description"),
        user_query=request.form.get("name"),
        image_of_recipe=image_cloudinary_url,  # now this is the URL of the image in Cloudinary
        share=True,
        generated_by_ia=False,
        user_id=user,
    )

    # Añadir y hacer commit a la nueva entrada
    db.session.add(new_recipe_chat)
    db.session.commit()

    # Retornar la receta, la URL de la imagen y el ID de la receta en la respuesta
    return jsonify({"recipe": request.form.get("user_query"), "image_url": image_cloudinary_url, "recipe_id": new_recipe_chat.id})



@rrecipe.route('/AllManuelRecipes', methods=['GET'])
@jwt_required()
def get_all_manual_recipes():

    jwt_claims = get_jwt()
    print(jwt_claims)
    user = jwt_claims["users_id"]
    print("el id del USUARIO:",user)

    manual_recipes = RecipeChat.query.filter_by(generated_by_ia=False, user_id=user).all()
    manual_recipes = list(map(lambda item: item.serialize(), manual_recipes))
    print(manual_recipes)

    return jsonify(manual_recipes), 200


@rrecipe.route('/AllShareRecipesManual', methods=['GET'])
@jwt_required()
def get_all_share_recipes_manual():

    jwt_claims = get_jwt()
    print(jwt_claims)
    user = jwt_claims["users_id"]
    print("el id del USUARIO:",user)

    share_recipes = Recipe.query.filter_by(share=True).all()
    share_recipes = list(map(lambda item: item.serialize(), share_recipes))
    print(share_recipes)

    return jsonify(share_recipes), 200















# @chat.route('/AddRecipe', methods=['POST'])
# @jwt_required()
# def add_recipe():

#     jwt_claims = get_jwt()
#     print(jwt_claims)
#     user = jwt_claims["users_id"]
#     print("el id del USUARIO:",user)

#     if 'image_of_recipe' not in request.files:
#         raise APIException("No image to upload")
#     if 'description' not in request.form:
#         raise APIException("No description to upload")
#     if 'user_query' not in request.form:
#         raise APIException("No user_query to upload")
    
#     # Consigue un timestamp y formatea como string
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     image_cloudinary_url = cloudinary.uploader.upload(
#         request.files['image_of_recipe'],
#         public_id = f'{request.form.get("user_query").replace(" ", "_")}_{timestamp}',
#     )

#     new_recipe_chat = RecipeChat(
#         name="nombre de la receta",  # actualiza esto
#         description=request.form.get("description"),
#         user_id=user, 
#         user_query=request.form.get("user_query"),
#         image_of_recipe=image_cloudinary_url,  # ahora esto es la URL de la imagen en Cloudinary
#         share=False,
#     )

#     # Añadir y hacer commit a la nueva entrada
#     db.session.add(new_recipe_chat)
#     db.session.commit()

#     # Retornar la receta, la URL de la imagen y el ID de la receta en la respuesta
#     return jsonify({"recipe": request.form.get("user_query"), "image_url": image_cloudinary_url, "recipe_id": new_recipe_chat.id})



#COMENTADO PARA PROBAR LA RUTA ADDRECIPE
# @chat.route('/AllShareRecipes', methods=['GET'])
# @jwt_required()
# def get_all_share_recipes():

#     jwt_claims = get_jwt()
#     print(jwt_claims)
#     user = jwt_claims["users_id"]
#     print("el id del USUARIO:",user)

#     share_recipes = RecipeChat.query.filter_by(share=True).all()
#     share_recipes = list(map(lambda item: item.serialize(), share_recipes))
#     print(share_recipes)

#     return jsonify(share_recipes), 200

# @chat.route('/EditRecipeChat', methods=['POST'])
# @jwt_required()
# def edit_recipe_chat():

#     jwt_claims = get_jwt()
#     print(jwt_claims)
#     user = jwt_claims["users_id"]
#     print("el id del USUARIO:",user)

#     id = request.form.get("id")
#     print("ID DE RECETA:", id)

#     if 'image_of_recipe' not in request.files:
#         raise APIException("No image to upload")
#     if 'description' not in request.form:
#         raise APIException("No description to upload")
#     if 'user_query' not in request.form:
#         raise APIException("No user_query to upload")
#     if 'id' not in request.form:
#         raise APIException("No id to upload")
    
#     # Consigue un timestamp y formatea como string
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     result = cloudinary.uploader.upload(
#         request.files['image_of_recipe'],
#         # public_id=f'recipe/{user.id}/{request.form.get("user_query")}',
#         # public_id=f'recipe/user/image_of_recipe',
#         public_id = f'{request.form.get("user_query").replace(" ", "_")}_{timestamp}',

#         #Para darle un tamaño específico a la imagen:
#         # crop='limit',
#         # width=450,
#         # height=450,
#         # eager=[{
#         #     'width': 200, 'height': 200,
#         #     'crop': 'thumb', 'gravity': 'face',
#         #     'radius': 100
#         # },
#         # ],
#         # tags=['profile_picture']
#     )

   

#     my_image = RecipeChat.query.get(id)
#     my_image.image_of_recipe = result['secure_url']
#     my_image.description = request.form.get("description")
#     my_image.user_query = request.form.get("user_query")
#     my_image.user_id = user
    
#     db.session.add(my_image) 
#     db.session.commit()

#     return jsonify(my_image.serialize()), 200


# @chat.route('/getChatHistory', methods=['GET'])
# @jwt_required()
# def get_chat_history():

#     jwt_claims = get_jwt()
#     print(jwt_claims)
#     user_id = jwt_claims["users_id"]
    
#     recipes = RecipeChat.query.filter_by(user_id=user_id).all()
#     recipes = list(map(lambda item: item.serialize(), recipes))
#     print(recipes)

#     return jsonify(recipes), 200

# @chat.route('/recipe', methods=['POST'])
# @jwt_required()
# def generate_recipe():

#     jwt_claims = get_jwt()
#     print(jwt_claims)
#     user_id = jwt_claims["users_id"]

#     print(user_id)

#     data = request.get_json()
#     prompt = "Eres una pagina web de recetas que responde con descripcion de la receta de una parráfo, una lista de ingredientes y un paso a paso para preparar la receta solicitada por el usuario: "+ data['prompt']

#     # Genera la receta
#     completion = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         n=1,
#         max_tokens=1024
#     )
#     recipe_text = completion.choices[0].text

#     # Genera la imagen
#     response = openai.Image.create(
#         prompt=data['prompt'],
#         n=1,
#         size="1024x1024"
#     )
#     image_url = response['data'][0]['url']

#     # Descarga la imagen
#     img_data = requests.get(image_url).content

#     # Consigue un timestamp y formatea como string
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Guarda la imagen en Cloudinary
#     upload_result = cloudinary.uploader.upload(
#         img_data,
#         public_id = f'{data["prompt"].replace(" ", "_")}_{timestamp}', 
#         resource_type = "auto" 
#     )
#     image_cloudinary_url = upload_result['url']

#     # Crear una nueva entrada en la base de datos
#     new_recipe_chat = RecipeChat(
#         name="nombre de la receta",  # actualiza esto
#         description=recipe_text,
#         user_id=user_id, 
#         user_query=data['prompt'],
#         image_of_recipe=image_cloudinary_url,  # ahora esto es la URL de la imagen en Cloudinary
#         share=False,
#     )

#     # Añadir y hacer commit a la nueva entrada
#     db.session.add(new_recipe_chat)
#     db.session.commit()

#     # Retornar la receta, la URL de la imagen y el ID de la receta en la respuesta
#     return jsonify({"recipe": recipe_text, "image_url": image_cloudinary_url, "recipe_id": new_recipe_chat.id})


# @chat.route('/ShareRecipeChat/<int:id>', methods=['PUT'])
# @jwt_required()
# def share_recipe_chat(id):
#     print("ID DE LA RECETA: ", id)

#     jwt_claims = get_jwt()
#     print(jwt_claims)
    
    

#    #comprobar si el user_id existe en las jwt_claims
#     if "users_id" not in jwt_claims:
#         return jsonify({"msg": "User not found"}), 401
#     user_id = jwt_claims["users_id"]
#     print("ID DEL USUARIO: ", user_id)

#     if user_id != jwt_claims["users_id"]:
#         return jsonify({"msg": "Unauthorized user"}), 401
  
#     recipe = RecipeChat.query.filter_by(user_id=user_id, id=id).first()

#     if recipe:
#         recipe.share = True
#         db.session.commit()
#         return jsonify(recipe.serialize()), 200
#     else:
#         raise APIException("Recipe not found", status_code=404)
    

# @chat.route('/UnShareRecipeChat/<int:id>', methods=['PUT'])
# @jwt_required()
# def unshare_recipe_chat(id):
#     print("ID DE LA RECETA: ", id)

#     jwt_claims = get_jwt()
#     print(jwt_claims)
    
    

#    #comprobar si el user_id existe en las jwt_claims
#     if "users_id" not in jwt_claims:
#         return jsonify({"msg": "User not found"}), 401
#     user_id = jwt_claims["users_id"]
#     print("ID DEL USUARIO: ", user_id)

#     if user_id != jwt_claims["users_id"]:
#         return jsonify({"msg": "Unauthorized user"}), 401
  
#     recipe = RecipeChat.query.filter_by(user_id=user_id, id=id).first()

#     if recipe:
#         recipe.share = False
#         db.session.commit()
#         return jsonify(recipe.serialize()), 200
#     else:
#         raise APIException("Recipe not found", status_code=404)













# #RUTAS ADICIONALES PARA EL CHATBOT: NO EN USO

# @chat.route('/chatgpt', methods=['POST'])
# def open_ai():
#     body =request.get_json()    
#     prompt = "Eres una pagina web de recetas que responde con descripcion de la receta, una lista de ingredientes y un paso a paso para preparar la receta solicitada por el usuario: "+ body['prompt']

#     completation = openai.Completion.create(engine="text-davinci-003",
#                             prompt=prompt,
#                             n=1,
#                             max_tokens=1024)
    
#     #print(completation.choices[0])
#     print(completation.choices[0].text)
#     response = {
#         "message":completation.choices[0].text
#     }
#     return jsonify(response), 200

# # MPT-7b : 64k tokens, ggml, q4_0, 128bits 4Q 
# # Oobaboonga, Koboldcpp

# @chat.route('/imageRecipe', methods=['POST'])
# def image_recipe():
#     data = request.get_json()
#     prompt = data.get('prompt', 'a white siamese cat')

#     response = openai.Image.create(
#         prompt=prompt,
#         n=1,
#         size="1024x1024"
#     )
#     image_url = response['data'][0]['url']

#     # Descarga la imagen
#     img_data = requests.get(image_url).content

#     # Consigue un timestamp y formatea como string
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Guarda la imagen en tu servidor (actualiza 'path/to/save/image' al directorio donde quieres guardar las imágenes)
#     image_path = os.path.join('src/front/img', f'{prompt.replace(" ", "_")}_{timestamp}.jpg')
#     with open(image_path, 'wb') as handler:
#         handler.write(img_data)

#     # Crear una nueva entrada en la base de datos
#     new_recipe_chat = RecipeChat(
#         name="nombre de la receta",  # actualiza esto
#         description="descripción de la receta",  # actualiza esto
#         user_id=1,  # actualiza esto
#         user_query=prompt,
#         image_of_recipe=image_path  # ahora esto es la ruta local de la imagen en tu servidor
#     )

#     # Añadir y hacer commit a la nueva entrada
#     db.session.add(new_recipe_chat)
#     db.session.commit()

#     # Retornar la ruta de la imagen y el ID de la receta en la respuesta
#     return jsonify({"image_path": image_path, "recipe_id": new_recipe_chat.id})

