import time
import os
from flask import Flask, jsonify, request
import requests
import base64
from flask_cors import CORS
from dotenv import load_dotenv
from recipe_type import cuisine_type, diet
import openai

load_dotenv()

integration_app = Flask(__name__)
CORS(integration_app, origins='http://localhost:3000')

IMAGE_CLASSIFICATION_URL = 'http://localhost:5001/classify_image'
RECIPE_GENERATOR_URL = 'http://localhost:5002'
DATABASE_URL = 'http://localhost:5004'
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')


def analyze_food_description(description):
	# Set up OpenAI API credentials
	openai.api_key = OPEN_AI_API_KEY

	# Define the prompt for the OpenAI completion
	prompt = f"analyze the food description: {description} and choose the most suitable Cuisine based on the options: {cuisine_type}, diet based on the options {diet} and Excluded_ingredients. Output in the format of \n\n Cuisine: \n\n Diet: \n\n Excluded Ingredients: \n\nIf you are unable to find a suitable option, please enter None."
	response = openai.Completion.create(
		engine="text-davinci-003",
		prompt=prompt,
		temperature=0.5,
		max_tokens=100,
		n=1,
		stop=None,
		timeout=10,
	)

	# Extract the generated text from the OpenAI response
	generated_text = response.choices[0].text.strip()
	print(generated_text)

	# Parse the generated text to extract the cuisine, diet type, and excluded ingredients
	cuisine = ""
	diet_type = ""
	excluded_ingredients = []

	lines = generated_text.split("\n")
	for line in lines:
		if line.startswith("Cuisine:"):
			cuisine = line.replace("Cuisine:", "").strip()
		elif line.startswith("Diet:"):
			diet_type = line.replace("Diet:", "").strip()
		elif line.startswith("Excluded Ingredients:"):
			excluded_ingredients = line.replace("Excluded Ingredients:", "").strip().split(",")

	print(diet_type.lower())
	# Check if the cuisine is valid
	if cuisine not in cuisine_type:
		cuisine = "None"

	# Check if the diet type is valid
	if diet_type.lower() not in diet:
		diet_type = "None"

	print("diet_type: ", diet_type)

	# String the excluded ingredients into a string
	excluded_ingredients = ",".join(excluded_ingredients)

	# Return the results as a dictionary
	return {
		"cuisine": cuisine,
		"diet_type": diet_type.lower(),
		"excluded": excluded_ingredients
	}


# Default route
@integration_app.route('/')
def index():
	return "Hello World! This is integration microservice."


# Post route to receive the image in the form of base64 string and return the recipe to the frontend
@integration_app.route('/get_recipe_from_image', methods=['POST'])
def get_recipe_from_image():
	try:
		# Get the image and description from the request
		image = request.json.get('image')
		description = request.json.get('description')

		# Make a request to the image classification service to get ingredients
		responses = requests.post(IMAGE_CLASSIFICATION_URL, json={"image": image})
		responses.raise_for_status()
		ingredients = responses.json().get('ingredients')
		predicted_classes = responses.json().get('predicted_classes')
		error = responses.json().get('error')

		if error != "None":
			# If the error field is not "None," return the error response from the image classification service
			return jsonify({"error": error, "message": "Error in classifying image."})

		# Make a post request to the recipe generator service to get the recipe
		analyzed_food_description = analyze_food_description(description)
		response_recipe = requests.post(f"{RECIPE_GENERATOR_URL}/generate_recipe",
										json={"ingredients": ingredients,
											  "predicted_classes": predicted_classes,
											  "cuisine": analyzed_food_description['cuisine'],
											  "diet_type": analyzed_food_description['diet_type'],
											  "excluded": analyzed_food_description['excluded']})
		response_recipe.raise_for_status()
		recipe = response_recipe.json()

		return recipe

	except requests.exceptions.RequestException as e:
		# Handle request-related exceptions
		return jsonify({"error": "Error in making requests.", "message": str(e)}), 500

	except Exception as e:
		# Handle other exceptions
		return jsonify({"error": "An error occurred.", "message": str(e)}), 500


# Post route to receive the recipe from the frontend and store it in the database using the database service
@integration_app.route('/save_recipe', methods=['POST'])
def save_recipe():
	user_id = request.json.get('user_id')
	recipe_data = request.json.get('recipe_data')
	collection = request.json.get('collection')
	recipe_name = request.json.get('recipe_name')

	try:
		# Make a post request to the database service to save the recipe
		response = requests.post(f"{DATABASE_URL}/save_recipe",
								 json={"user_id": user_id, "recipe_data": recipe_data, "collection": collection,
									   "recipe_name": recipe_name})
		response.raise_for_status()
		return response.json()

	except requests.exceptions.RequestException as e:
		# Handle request-related exceptions
		return jsonify({"error": "Error in making requests.", "message": str(e)}), 500

	except Exception as e:
		# Handle other exceptions
		return jsonify({"error": "An error occurred.", "message": str(e)}), 500


# Get route to get the recipes from the database using the database service and return it to the frontend
@integration_app.route('/get_recipes/<user_id>/', methods=['GET'])
def get_recipes(user_id):
	try:
		# Make a get request to the database service to get the recipes
		response = requests.get(f"{DATABASE_URL}/get_recipes/{user_id}")
		response.raise_for_status()
		return response.json()

	except requests.exceptions.RequestException as e:
		# Handle request-related exceptions
		return jsonify({"error": "Error in making requests.", "message": str(e)}), 500

	except Exception as e:
		# Handle other exceptions
		return jsonify({"error": "An error occurred.", "message": str(e)}), 500


# Get single recipe from the database using the database service and return it to the frontend
@integration_app.route('/get_recipe/<recipe_name>/', methods=['GET'])
def get_recipe(recipe_name):
	try:
		# Make a get request to the database service to get the recipes
		response = requests.get(f"{DATABASE_URL}/get_recipe/{recipe_name}")
		response.raise_for_status()
		return response.json()

	except requests.exceptions.RequestException as e:
		# Handle request-related exceptions
		return jsonify({"error": "Error in making requests.", "message": str(e)}), 500

	except Exception as e:
		# Handle other exceptions
		return jsonify({"error": "An error occurred.", "message": str(e)}), 500


if __name__ == '__main__':
	integration_app.run(debug=True, host='localhost', port=5003)
