from flask import Blueprint, request, jsonify
import openai
from flask_pymongo import PyMongo
from app import app, mongo  # Import app and mongo from the app module

# Initialize the Blueprint
routes_bp = Blueprint('routes', __name__)

# Route for chatgpt
@routes_bp.route('/get_response', methods=['POST'])
def chatgpt():
    try:
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

        reply = {
            "user_input": user_input,
            "response": response.choices[0].text
        }

        # Extract and return the response text
        return jsonify(reply), 201
    except Exception as e:
        # Handle OpenAI API errors or any other exceptions
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/get-history', methods=['GET'])
def get_history():
    # Query the MongoDB collection named 'test' to fetch data
    userd = mongo.db.test.find({})

    # Serialize the retrieved data
    serialized_user_data = []
    for user in userd:
        user['_id'] = str(user['_id'])
        serialized_user_data.append(user)
    return jsonify(serialized_user_data), 201

@routes_bp.route('/save-history', methods=['POST'])
def save_history():
    try:
        # Get JSON data from the request
        response_data = request.get_json()
        if not response_data:
            print("Error: The response_data list is empty.")
        else:
            # Proceed with processing the data
            json_dict = {'chatd': response_data}
            print(json_dict)
            mongo.db.test.insert_one(json_dict)
            # Return a JSON response indicating success
        return jsonify({'message': 'Data inserted successfully'}), 201
    except Exception as e:
        # Handle exceptions (e.g., invalid JSON, database errors)
        return jsonify({'error': str(e)}), 500

# Blueprint registration should come after all routes are defined
app.register_blueprint(routes_bp)
