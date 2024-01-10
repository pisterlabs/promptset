"""
This module takes care of starting the API Server, Loading the DB and Adding the endpoints
"""
from flask import Flask, request, jsonify, url_for, Blueprint
from api.models import db, User, Recipe, Category, TokenBlockedList, Favorite
from api.utils import generate_sitemap, APIException
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from flask_bcrypt import Bcrypt
import openai

api = Blueprint('api', __name__)
app = Flask(__name__)
bcrypt = Bcrypt(app)

# Register an user
@api.route('/signup', methods=['POST'])
def user_create():
    data = request.get_json()
    new_user = User.query.filter_by(email=data["email"]).first()
    if(new_user is not None):
        return jsonify({
            "message": "Registered user"
        }), 400
    secure_password = bcrypt.generate_password_hash(data["password"], rounds=None).decode("utf-8")
    new_user = User(email=data["email"], password=secure_password,
                    first_name=data["first_name"], last_name=data["last_name"],
                    is_active=True, security_question=data["security_question"],
                    security_answer=data["security_answer"], is_admin=data["is_admin"])
    db.session.add(new_user)
    db.session.commit()
    return jsonify(new_user.serialize()), 201

# Allow to login into the application
@api.route('/login', methods=['POST'])
def user_login():
    user_email = request.json.get("email")
    user_password = request.json.get("password")
    user = User.query.filter_by(email=user_email).first()
    if(user is None):
        return jsonify({
            "message": "User not found"
        }), 401
    # verify password
    if not bcrypt.check_password_hash(user.password, user_password):
        return jsonify({"message": "Wrong password"}), 401
    
    # generate token
    access_token = create_access_token(identity = user.id)
    return jsonify({"accessToken": access_token, "id": user.id})

# Allow to logout into the application
@api.route('/logout', methods=['POST'])
@jwt_required()
def user_logout():
    jwt = get_jwt()["jti"]
    tokenBlocked = TokenBlockedList(jti = jwt)
    db.session.add(tokenBlocked)
    db.session.commit()
    return jsonify({"message": "Token revoked"})

# Recovery the password
@api.route('/passwordRecovery', methods=['PUT'])
def user_password_recovery():
    user_email = request.json.get("email")
    user_new_password = request.json.get("new_password")
    user_security_question = request.json.get("security_question")
    user_security_answer = request.json.get("security_answer")
    user = User.query.filter_by(email=user_email).first()
    if(user is None):
        return jsonify({
            "message": "User not found"
        }), 401
    # verify security question and security answer
    if not (user_security_question == user.security_question and user_security_answer == user.security_answer):
        return jsonify({"message": "Security question and answer do not match"}), 401
    
    # change password
    secure_password = bcrypt.generate_password_hash(user_new_password, rounds=None).decode("utf-8")
    user.password = secure_password

    db.session.commit()
    return jsonify(user.serialize()), 200

# Show all users
@api.route("/allUsers", methods=["GET"])
@jwt_required()
def user_show_all():
    get_users = User.query.all()
    user_list = list(map(lambda u : u.serialize(), get_users))
    return jsonify({"users": user_list})

# Show a single user by ID
@api.route('/showUser/<int:userId>', methods=['GET'])
@jwt_required()
def user_show_by_id(userId):
    user = User.query.filter_by(id=userId).first()
    if(user is None):
        return jsonify({
            "message": "User does not exist"
        }), 400
    return jsonify({"user": user.serialize()}), 200

# Edit an user by ID
@api.route("/updateUser/<int:userId>", methods=["PUT"])
@jwt_required()
def user_update(userId):
    first_name = request.json['first_name']
    last_name = request.json['last_name']
    is_active = request.json['is_active']
    is_admin = request.json['is_admin']
    security_question = request.json['security_question']
    security_answer = request.json['security_answer']
    user = User.query.get(userId)
    if(user is None):
        return jsonify({
            "message": "User not found"
        }), 400
    user.first_name = first_name
    user.last_name = last_name
    user.is_active = is_active
    user.is_admin = is_admin
    user.security_question = security_question
    user.security_answer = security_answer

    db.session.commit()

    return jsonify(user.serialize()), 200

# Deactivate an user by ID
@api.route("/deleteUser/<int:userId>", methods=["DELETE"])
@jwt_required()
def user_delete(userId):
    user = User.query.get(str(userId))
    if(user is None):
        return jsonify({
            "message": "User not found"
        }), 400
    db.session.delete(user)
    db.session.commit()

    return jsonify(user.serialize()), 200

# Add a new category
@api.route('/addCategory', methods=['POST'])
@jwt_required()
def category_create():
    data = request.get_json()

    new_category = Category(
        name=data["name"], description=data["description"]
    )
    db.session.add(new_category)
    db.session.commit()
    return jsonify(new_category.serialize()), 201

# Edit a category by ID
@api.route('/updateCategory/<int:categoryId>', methods=['PUT'])
@jwt_required()
def category_update(categoryId):
    name = request.json['name']
    description = request.json['description']

    category = Category.query.get(categoryId)
    if(category is None):
        return jsonify({
            "message": "Category not found"
        }), 400
    category.name = name
    category.description = description


    db.session.commit()

    return jsonify(category.serialize()), 200

# Delete a category by ID
@api.route("/deleteCategory/<int:categoryId>", methods=["DELETE"])
@jwt_required()
def category_delete(categoryId):
    category = Category.query.get(str(categoryId))
    if(category is None):
        return jsonify({
            "message": "Category not found"
        }), 400
    db.session.delete(category)
    db.session.commit()

    return jsonify(category.serialize()), 200

# Show the all categories
@api.route('/showCategories', methods=['GET'])
@jwt_required()
def category_show_all():
    get_categories = Category.query.all()
    category_list = list(map(lambda c : c.serialize(), get_categories))
    return jsonify({"categories": category_list})

# Show a single category by ID
@api.route('/showCategory/<int:categoryId>', methods=['GET'])
@jwt_required()
def category_show_by_id(categoryId):
    category = Category.query.filter_by(id=categoryId).first()
    if(category is None):
        return jsonify({
            "message": "Category does not exist"
        }), 400
    return jsonify({"category": category.serialize()}), 200

# Show the all recipes
@api.route('/showRecipes', methods=['GET'])
@jwt_required()
def recipes_all_show():
    recipes = Recipe.query.all()
    dictionary_recipes = list(map(lambda r : r.serialize(), recipes))
    return jsonify({"recipes": dictionary_recipes}), 200

# Show a single recipe by ID
@api.route('/showRecipe/<int:recipeId>', methods=['GET'])
@jwt_required()
def recipe_show_by_id(recipeId):
    recipe = Recipe.query.filter_by(id=recipeId).first()
    if(recipe is None):
        return jsonify({
            "message": "Recipe does not exist"
        }), 400
    return jsonify({"recipe": recipe.serialize()}), 200

# Show the all recipes into a specific category by ID
@api.route('/showRecipes/<int:categoryId>', methods=['GET'])
@jwt_required()
def recipes_by_category_show(categoryId):
    recipes = Recipe.query.filter_by(category_id=categoryId).all()
    if(recipes is None):
        return jsonify({
            "message": "Recipe does not exist with this category"
        }), 400
    dictionary_recipes = list(map(lambda r : r.serialize(), recipes))
    return jsonify({"recipes": dictionary_recipes}), 201

# Edit a specific recipe by ID
@api.route('/updateRecipe/<int:recipeId>', methods=['PUT'])
@jwt_required()
def recipe_update(recipeId):
    name = request.json['name']
    description = request.json['description']
    is_active = request.json['is_active']
    elaboration = request.json['elaboration']
    image = request.json['image']
    category_id = request.json['category_id']
    updated_recipe = Recipe.query.filter_by(id=recipeId).first()
    if(updated_recipe is None):
        return jsonify({
            "message": "Recipe does not exist"
        }), 400
    
    updated_recipe.name = name
    updated_recipe.description = description
    updated_recipe.is_active = is_active
    updated_recipe.elaboration = elaboration
    updated_recipe.image = image
    updated_recipe.category_id = category_id
    db.session.commit()
    return jsonify(updated_recipe.serialize()), 200

# Add recipe with the information
@api.route('/addRecipe', methods=['POST'])
@jwt_required()
def recipe_create():
    data = request.get_json()

    new_recipe = Recipe(
        name=data["name"], description=data["description"], is_active=True,
        elaboration=data["elaboration"], image=data["image"], category_id=data["category_id"],
        user_id=data["user_id"]
    )
    db.session.add(new_recipe)
    db.session.commit()
    return jsonify(new_recipe.serialize()), 201

# Delete a recipe by recipe Id
@api.route("/deleteRecipe/<int:recipeId>", methods=["DELETE"])
@jwt_required()
def recipe_delete(recipeId):
    recipe = Recipe.query.get(recipeId)
    if(recipe is None):
        return jsonify({
                "message": "Recipe does not exist"
            }), 400
    db.session.delete(recipe)
    db.session.commit()

    return jsonify(recipe.serialize()), 200

# Show all recipes in favorites
@api.route('/showRecipesFavorites', methods=['GET'])
@jwt_required()
def recipes_all_favorites_show():
    favorites = Favorite.query.join(Recipe).join(User).all()
    dictionary_recipes = list(map(lambda f : f.serialize(), favorites))
    return jsonify({"favorites": dictionary_recipes}), 200

# Show all recipes in favorites by user Id
@api.route('/showRecipesFavoritesbyUserId', methods=['GET'])
@jwt_required()
def recipes_all_favorites_by_userId_show():
    user_id = get_jwt_identity()
    favorites = Favorite.query.filter_by(user_id = user_id)
    dictionary_recipes = list(map(lambda f : f.serialize(), favorites))
    return jsonify({"favorites": dictionary_recipes}), 200

# Add a recipe to favorite
@api.route('/addRecipeToFavorite/<int:recipeId>/', methods=['POST'])

@jwt_required()
def favorite_add_recipe(recipeId):
    user_id = get_jwt_identity()
    recipe = Recipe.query.get(recipeId)
    if(recipe is None):
        return jsonify({
                "message": "Recipe does not exist"
            }), 400
    recipe_favorite = Favorite.query.filter_by(recipe_id = recipeId).filter_by(user_id = user_id).first()
    if(recipe_favorite is not None):
        return jsonify({
                "message": "Recipe already exist in favorites"
            }), 400
    new_favorite = Favorite(
        recipe_id=recipeId, user_id=user_id
    )
    db.session.add(new_favorite)
    db.session.commit()
    return jsonify({"message": "Recipe added to favorites"}), 201

# Delete recipe from favorites by recipeId
@api.route("/deleteRecipeFromFavorites/<int:recipeId>", methods=["DELETE"])
@jwt_required()
def recipe_delete_from_favorites(recipeId):
    user_id = get_jwt_identity()
    recipe_favorite = Favorite.query.filter_by(recipe_id = recipeId).filter_by(user_id = user_id).first()
    if(recipe_favorite is None):
        return jsonify({
                "message": "Recipe does not exist"
            }), 400
    db.session.delete(recipe_favorite)
    db.session.commit()

    return jsonify({"message": "Recipe deleted from favorites"}), 200

@api.route('/call-chatGPT', methods=['GET'])
def generateChatResponse(prompt):
    return call_chatGPTApi(prompt)

'''@api.route('/helloprotected', methods=['GET'])
@jwt_required()
def hello_protected_get():
    user_id = get_jwt_identity()
    return jsonify({"userId": user_id, "msg": "hello protected route"})'''


def call_chatGPTApi(prompt):
    messages = []
    messages.append({"role": "system", "content": "Your name is Karabo. You are a helpful assistant."})
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'
    return answer