"""
This module interfaces with user data.
"""

# import data.food
import os
import random
from string import ascii_uppercase
import data.db_connect as con
from PIL import Image
import pytesseract
import datetime
import openai
from google.oauth2 import id_token
from google.auth.transport import requests
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import data.food as fd

TEST_USERNAME_LENGTH = 6
TEST_NAME_LENGTH = 6

NAME = 'Name'
PANTRY = 'Pantry'
USERNAME = "Username"
SAVED_RECIPES = 'Saved_Recipes'
INSTACART_USR = 'Instacart_User_Info'
GROCERY_LIST = 'Grocery List'
ALLERGENS = 'Allergens'
AUTH_EXPIRES = "Auth_Exp"
AUTH_TYPE = "Auth_Type"
PASSWORD = "Password"
REFRESH_TOKEN = 'Refresh_Token'
STREAK = "Streak"


class AuthTokenExpired(Exception):
    pass


def _get_test_username():
    con.connect_db()
    username = ''.join(random.choices(ascii_uppercase, k=TEST_USERNAME_LENGTH))
    while user_exists(username):
        username = ''.join(random.choices(
            ascii_uppercase, k=TEST_USERNAME_LENGTH))
    return username


def _get_test_name():
    con.connect_db()
    name = ''.join(random.choices(ascii_uppercase, k=TEST_NAME_LENGTH))

    return name


def generate_exp():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=1)


def _get_test_auth_token(username="TESTING"):
    return {
        'email': username,
        'name': username,
        'exp': int(generate_exp().timestamp())
    }


def user_exists(username):
    con.connect_db()
    try:
        con.fetch_one(con.USERS_COLLECTION, {USERNAME: username})
        res = True
    except ValueError:
        res = False
    return res


def _create_test_google_user():
    username = _get_test_username()
    name = _get_test_name()
    exp = generate_exp()
    print(username, name, exp)
    test_user = create_user(username, name, exp)
    return test_user


def _create_test_user():
    username = _get_test_username()
    name = _get_test_name()
    password = "TEST_PASSWORD"
    register_user(username, name, password)
    return username


def _create_test_patch_user():
    username = "TEST"
    name = "TEST"
    auth_type = "Google"
    password = "password"
    new_user = {
        USERNAME: username,
        NAME: name,
        PANTRY: [],
        SAVED_RECIPES: [],
        INSTACART_USR: None,
        GROCERY_LIST: [],
        ALLERGENS: [],
        AUTH_TYPE: auth_type,
        AUTH_EXPIRES: 0,
        PASSWORD: password,
        STREAK: 0
    }
    return new_user


def get_users():
    con.connect_db()
    users = con.fetch_all(con.USERS_COLLECTION)
    for user in users:
        user[con.MONGO_ID] = str(user[con.MONGO_ID])
    return users


def get_user(username: str) -> str:
    con.connect_db()
    if auth_expired(username):
        raise AuthTokenExpired("User's Authentication Token is expired")
    try:
        res = con.fetch_one(con.USERS_COLLECTION, {USERNAME: username})
        res[con.MONGO_ID] = str(res[con.MONGO_ID])
    except ValueError:
        raise ValueError(f'User {username} does not exist')

    return res


def auth_expired(username: str) -> bool:
    exp = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {AUTH_EXPIRES: 1, con.MONGO_ID: 0}
    )

    return exp[AUTH_EXPIRES] <= datetime.datetime.utcnow().timestamp()


def valid_authentication(google_id_token):
    # Add check for CLIENT ID for app that accesses authentication
    # Maybe save valid CLIENT ID to check against in os.environ()
    idinfo = id_token.verify_oauth2_token(google_id_token, requests.Request())

    # aud = idinfo['aud']
    # if os.environ.get("GOOGLE_CLIENT_ID") != aud:
    #     raise ValueError("Invalid Token")

    exp = idinfo['exp']
    if exp < datetime.datetime.utcnow().timestamp():
        raise AuthTokenExpired("Expired token")
    return idinfo


def generate_refresh_token(username):
    # Set the expiration time, e.g., 30 days from now
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(days=30)

    refresh_token = generate_jwt(username, expiration_time.timestamp())

    return refresh_token


def refresh_user_token(refresh_token):
    con.connect_db()
    try:

        payload = jwt.decode(
            refresh_token,
            key=os.environ.get("JWT_SECRET_KEY"),
            algorithms='HS256',
            verify=True
        )
        print(payload)
        if datetime.datetime.utcnow().timestamp() > payload[AUTH_EXPIRES]:
            raise ValueError("Refresh Token Expired")
        print("HIIII")

        stored_refresh_token = con.fetch_one(
            con.USERS_COLLECTION,
            {USERNAME: payload[USERNAME]},
            {REFRESH_TOKEN: 1, con.MONGO_ID: 0}
        )[REFRESH_TOKEN]
        print(f'TOKENS: {refresh_token}: {stored_refresh_token}')

        if refresh_token != stored_refresh_token:
            raise ValueError("Invalid Refresh Token")

        exp = generate_exp().timestamp()
        con.update_one(
            con.USERS_COLLECTION,
            {USERNAME: payload[USERNAME]},
            {"$set": {AUTH_EXPIRES: exp}}
        )
        return generate_jwt(payload[USERNAME], exp)
    except jwt.ExpiredSignatureError:
        raise ValueError("Refresh Token Expired")
    except jwt.InvalidTokenError as e:
        print(str(e))
        raise ValueError("Invalid Refresh Token")


def auth_user(token):
    auth_user_google(token)


def auth_user_google(google_id_token):
    con.connect_db()
    try:
        id_info = valid_authentication(google_id_token)
        username = id_info['email']
        if not user_exists(username):
            raise ValueError("User associated with token does not exist")

        exp = int(id_info['exp'])

        con.update_one(
            con.USERS_COLLECTION,
            {USERNAME: username},
            {"$set": {AUTH_EXPIRES: exp}}
        )

    except ValueError as ex:
        # Invalid token
        raise ex
    except AuthTokenExpired as ex:
        raise ex


def generate_google_user(google_id_token):
    id_info = valid_authentication(google_id_token)

    username = id_info['email']
    name = id_info['name']
    exp = int(id_info['exp'])
    print(username, name, exp)
    create_user(username, name, exp)


def create_user(username: str, name: str,
                expires: datetime.datetime, password=None,
                refresh_token=None) -> dict:
    con.connect_db()
    if len(username) < 5:
        raise ValueError(f'Username {username} is too short')

    if user_exists(username):
        raise ValueError(f'User {username} already exists')

    auth_type = "Google" if password is None else "Self"

    new_user = {
        USERNAME: username,
        NAME: name,
        PANTRY: [],
        SAVED_RECIPES: [],
        INSTACART_USR: None,
        GROCERY_LIST: [],
        ALLERGENS: [],
        AUTH_TYPE: auth_type,
        AUTH_EXPIRES: int(expires.timestamp()),
        PASSWORD: password,
        STREAK: 0,
        REFRESH_TOKEN: refresh_token
    }

    add_ret = con.insert_one(con.USERS_COLLECTION, new_user)
    print(f'{add_ret}')
    return new_user


def remove_user(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    del_res = con.del_one(con.USERS_COLLECTION, {USERNAME: username})

    print(f'{del_res}')
    return f'Successfully deleted {username}'


def login_user(username, password):
    print(f'{username}')
    con.connect_db()
    if not user_exists(username):
        print('user_Exists')
        raise ValueError(f'User {username} does not exist')

    user_password_obj = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {PASSWORD: 1, con.MONGO_ID: 0}
    )
    user_password = user_password_obj[PASSWORD]

    if not check_password_hash(user_password, password):
        print("password")
        raise ValueError('Password does not match')

    exp = int(generate_exp().timestamp())
    print(exp)
    access_token = generate_jwt(username, exp)
    refresh_token = generate_refresh_token(username)

    print(access_token)

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$set": {AUTH_EXPIRES: exp, REFRESH_TOKEN: refresh_token}}
    )
    print("JSDFKLSJD")
    return access_token, refresh_token


def logout_user(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$set": {AUTH_EXPIRES: 0}}
    )


def generate_jwt(username, exp):

    # Create the JWT payload
    payload = {
        USERNAME: username,
        AUTH_EXPIRES: exp
    }

    # Encode the JWT
    # Run openssl rand -base64 12 to generate password
    token = jwt.encode(
        payload,
        os.environ.get("JWT_SECRET_KEY"),
        algorithm='HS256'
    )
    return token


def register_user(username, name, password):
    hashed_password = generate_password_hash(password, method='scrypt')
    exp = generate_exp()
    token = generate_jwt(username, exp.timestamp())
    refresh_token = generate_refresh_token(username)

    create_user(username, name, exp, hashed_password, refresh_token)

    return token, refresh_token


def get_pantry(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    pantry_res = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {PANTRY: 1, con.MONGO_ID: 0}
    )

    return pantry_res[PANTRY]


def add_to_pantry(username: str, food) -> str:
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")
    print(food)

    new_pantry_entries = [fd.get_food(
        ingredient[fd.INGREDIENT],
        ingredient[fd.QUANTITY],
        ingredient[fd.UNITS]
        ) for ingredient in food]

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$push": {PANTRY: {"$each": new_pantry_entries}}}
    )
    return f'Successfully added {food}'


def get_recipes(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    recipes_res = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {SAVED_RECIPES: 1, con.MONGO_ID: 0}
    )

    return recipes_res[SAVED_RECIPES]


def generate_recipe(username, query):
    app_key = '274c6a9381c49bc303a30cebb49c84d4'
    app_id = '29bf3511'
    query_string = 'https://api.edamam.com/api/recipes\
        /v2?type=public&q=' + query + '&app_id=' + app_id +\
        '&app_key=' + app_key
    x = requests.get(query_string)
    return x  # return full recipe response body


def generate_recipe_gpt(username, query):   # generate recipe with AI
    openai.api_key = os.environ.get("OPENAI_KEY")
    prompt = f"Based on the following requirements,\
          please recommend a recipe:\n\n{query}\n\nRecipe:"
    # Make the API call
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=200  # You can adjust this value based on your needs
    )
    # Extract the recommended recipe from the response
    recommended_recipe = response.choices[0].text.strip()
    return recommended_recipe


def add_to_recipes(username, recipe):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$push": {SAVED_RECIPES: recipe}}
    )

    return f'Successfully added {recipe}'


def delete_recipe(username, recipe):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$pull": {SAVED_RECIPES: recipe}}
    )

    return f'Successfully deleted {recipe}'


def get_streak(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    recipes_res = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {STREAK: 1, con.MONGO_ID: 0}
    )

    return recipes_res[STREAK]


def inc_streak(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$inc": {STREAK: 1}}
    )

    return 'Successfully incremented streak counter'


def recognize_receipt(username: str, image_path=None, image=None):
    openai.api_key = os.environ.get("OPENAI_KEY")
    if (image_path and not image):
        # Load the image from the specified path
        image = Image.open(image_path)
    elif (not image):  # neither the path nor image is provided
        return None
    # Perform OCR using pytesseract
    ocr_text = pytesseract.image_to_string(image)
    # Print or save the extracted text
    print(ocr_text)
    prompt = f"Extract pantry items from the following text: {ocr_text}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=200
    )
    # Extract the generated text from ChatGPT's response
    generated_text = response.choices[0].text.strip()
    # Split the generated text into individual pantry items
    pantry_items = generated_text.split('\n')
    # Remove any empty or whitespace-only items
    pantry_items = [item.strip() for item in pantry_items if item.strip()]
    for food in pantry_items:
        add_to_pantry(username, food)
    return pantry_items
