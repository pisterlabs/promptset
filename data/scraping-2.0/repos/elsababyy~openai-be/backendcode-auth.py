from flask import Flask, request, jsonify,session, abort, redirect, request
from flask_cors import CORS
import requests
from flask_pymongo import PyMongo
import json
import pathlib
import os
# from dotenv import load_dotenv, find_dotenv
import openai
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
import json
from flask import Flask 
from flask.wrappers import Response
from flask.globals import request, session
import requests
from werkzeug.exceptions import abort
from werkzeug.utils import redirect
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import os, pathlib
import google
import jwt
from flask_cors import CORS

os.environ["OPENAI_API_KEY"] = "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"
# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['Access-Control-Allow-Origin'] = '*'
app.config["Access-Control-Allow-Headers"]="Content-Type"
app.config["MONGO_URI"] = "mongodb://localhost:27017/openai"
mongo = PyMongo(app)
app.secret_key = "CodeSpecialist.com"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
GOOGLE_CLIENT_ID = "990487476652-dct8v0k5a9l9776ci05g3dq15p8muj33.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")
 

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=[
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ],
    redirect_uri="http://127.0.0.1:5000/callback",
)


# wrapper
def login_required(function):
    def wrapper(*args, **kwargs):
        encoded_jwt=request.headers.get("Authorization").split("Bearer ")[1] #extract the actual token
        if encoded_jwt==None: #no JWT was found in the "Authorization" header.
            return abort(401)
        else:
            return function()
    return wrapper


def Generate_JWT(payload):#payload: dictionary containing the data wanted to include in the JWT
    encoded_jwt = jwt.encode(payload, app.secret_key, algorithm='HS256') #acommonly used algorithm which ensures that the token is securely signed.
    return encoded_jwt


@app.route("/callback")
def callback():
    #Fetch the access token and credentials using the authorization response
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials

    #Create a requests session and a token request
    request_session = requests.session()
    token_request = google.auth.transport.requests.Request(session=request_session)

    #Verify the ID token obtained from Google
    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token, request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    #Store the user's Google ID in the session
    session["google_id"] = id_info.get("sub")
    
    #removing the specific audience, as it is throwing error
    del id_info['aud']

    #Generate a JWT token using the ID token's payload data
    jwt_token=Generate_JWT(id_info)

    #Prepare user data for insertion into a MongoDB collection
    data={
        'name':id_info.get('name'),
        'email':id_info.get('email'),
        'picture':id_info.get('picture')
    }

    #Insert the user data into a MongoDB collection named 'users'
    mongo.db.users.insert_one(data)

    #Redirect the user to a specific URL with the JWT token as a query parameter
    return redirect(f"http://localhost:5173/chat?jwt={jwt_token}")
   

@app.route("/auth/google")
def login():

    #Generate the authorization URL and state
    authorization_url, state = flow.authorization_url()
    # Store the state so the callback can verify the auth server response.
    session["state"] = state
    return Response(
        response=json.dumps({'auth_url':authorization_url}),
        status=200,
        mimetype='application/json'
    )


@app.route("/logout")
def logout():
    #clear the local storage from frontend
    session.clear()
    return Response(
        response=json.dumps({"message":"Logged out"}),
        status=202,
        mimetype='application/json'
    )


@app.route("/home")
@login_required
def home_page_user():

    #Extract the JWT token from the "Authorization" header
    encoded_jwt=request.headers.get("Authorization").split("Bearer ")[1]

    #Attempt to decode and verify the JWT token
    try:
        decoded_jwt=jwt.decode(encoded_jwt, app.secret_key, algorithms=['HS256',])
        print(decoded_jwt)
    except Exception as e: 

        #Return an error response if JWT decoding fails
        return Response(
            response=json.dumps({"message":"Decoding JWT Failed", "exception":e.args}),
            status=500,
            mimetype='application/json'
        )

        #Return a JSON response containing the decoded JWT payload
    return Response(
        response=json.dumps(decoded_jwt),
        status=200,
        mimetype='application/json')


@app.route('/get_response', methods=['POST'])
def chatgpt():
    # Get data from the request
    data = request.get_json()
    user_input = data.get("user_input")  # Extract the user input from the JSON data
    # Create a response using the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use "text-davinci-003" as the engine
        prompt=user_input,
        max_tokens=3000,  # Limit the response to a certain number of tokens
        temperature=0.7  # Adjust the temperature parameter
    )
    reply={
            "user_input":user_input,
            "response":response.choices[0].text
        }
    # Extract and return the response text
    return jsonify(reply),201

@app.route('/get-history',methods=['GET'])
def get_history():

    #Query the MongoDB collection named 'test' to fetch data
    userd = mongo.db.test.find({})

    #Serialize the retrieved data
    serialized_user_data = []
    for user in userd:
        user['_id'] = str(user['_id'])
        serialized_user_data.append(user)
    return jsonify(serialized_user_data), 201

@app.route('/save-history',methods=['POST'])
def save_history():
    try:
        # Get JSON data from the request
        response_data = request.get_json()
        if not response_data:
            print("Error: The response_data list is empty.")
        else:
    # Proceed with processing the data
            json_dict = {'chatd':response_data}
            print(json_dict)
            mongo.db.test.insert_one(json_dict)
            # Return a JSON response indicating success
        return jsonify({'message': 'Data inserted successfully'}), 201
    except Exception as e:
        # Handle exceptions (e.g., invalid JSON, database errors)
        return jsonify({'error': str(e)}), 500
 

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)