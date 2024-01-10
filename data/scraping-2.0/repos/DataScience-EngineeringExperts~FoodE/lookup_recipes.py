import os
import requests
import random
import json
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import openai
import logging

# Load environment variables
load_dotenv()

# Creds needed for functions
mongo_conn_string = os.getenv('mongo_conn_string')
EDAMAM_APP_ID = os.getenv('EDAMAM_APP_ID')
EDAMAM_API_KEY = os.getenv('EDAMAM_API_KEY')
openai_api_key = os.getenv('openai_api_key')
EDAMAM_API_ENDPOINT = "https://api.edamam.com/search"
webhook_url = os.getenv('webhook_url')


# OpenAI API Configuration
openai.api_key = openai_api_key

# MongoDB Setup
client = MongoClient(mongo_conn_string)
db = client.menumaker
recipes_collection = db.recipes
results_collection = db.results
user_summary_collection = db.user_summary
log_collection = db.logs

class MongoDBHandler(logging.Handler):
    def emit(self, record):
        # Format the log record using the formatter
        log_entry = self.format(record)

        # Create a log document with a 'datetime' field
        log_document = {
            "log": log_entry,
            "datetime": datetime.now()  # Add the current datetime
        }

        # Insert the log document into MongoDB
        log_collection.insert_one(log_document)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a MongoDB log handler
mongo_handler = MongoDBHandler()
logger.addHandler(mongo_handler)

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mongo_handler.setFormatter(formatter)


# Direct/Explicit Tags called from Edamame
HealthLabel = [
    "fat-free", "low-fat-abs", "sugar-conscious", "low-sugar", "low-potassium", "kidney-friendly",
    "keto-friendly", "plant-based", "vegan", "vegetarian", "pescatarian", "paleo", "specific-carbs",
    "Mediterranean", "DASH", "dairy-free", "gluten-free", "wheat-free", "egg-free", "milk-free", "peanut-free",
    "tree-nut-free", "soy-free", "fish-free", "shellfish-free", "pork-free", "red-meat-free", "crustacean-free",
    "celery-free", "mustard-free", "sesame-free", "lupine-free", "mollusk-free", "alcohol-free", "no-oil-added",
    "no-sugar-added", "sulfite-free", "fodmap-free", "kosher", "alcohol-cocktail", "immuno-supportive"
]

DietLabel = [
    "balanced", "high-protein", "high-fiber", "low-fat", "low-carb", "low-sodium"
]

CautionLabel = [
    "gluten", "wheat", "eggs", "milk", "peanuts", "tree-nuts", "soy", "fish", "shellfish", "sulfites", "fodmap"
]

def fetch_and_format_recipes(username, timestamp):
    """
    Fetches the latest recipes for the given user and timestamp, and formats them for a Teams message.
    """
    # Fetch the latest recipes for the user
    recipes = recipes_collection.find({'username': username, 'timestamp': timestamp})

    # Format the recipes into a readable message
    message_lines = [f"Latest recipe details for {username}:"]
    for recipe in recipes:
        recipe_name = recipe.get('name', 'No recipe name')
        recipe_type = recipe.get('type', 'Unknown type')
        message_lines.append(f" - {recipe_type.capitalize()}: {recipe_name}")

    return '\n'.join(message_lines)


def send_teams_notification(message, webhook_url):
    """
    Sends a notification message to a Microsoft Teams channel.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Message successfully sent to Teams: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message to Teams: {e}")


# Valid Edamam health labels
valid_health_labels = set(HealthLabel + DietLabel + CautionLabel)

def determine_applicable_tags(user_summary):
    # Convert the user_summary including ObjectId to a string
    user_summary_str = json.dumps(user_summary, default=str)

    # Use OpenAI's model to determine applicable tags
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an AI chef, nutritionist, and certified dietitian. Use knowledge from industry leaders and "
                                "the top achievers. Retain and utilize their methods of success as well as discern patterns, "
                                "strategies, habits that contribute to their high success. Your assignment is to find "
                                f"applicable food tags from the {valid_health_labels} library. "
                                "Return the filtered valid health labels in JSON format. Return the labels for the "
                                "entire group without specifying the person to which it belongs. "},
            {"role": "user", "content": user_summary_str}
        ]
    )
    applicable_tags = json.loads(response.choices[0].message['content'])
    # print(response)
    return [tag for tag in applicable_tags if tag in valid_health_labels]

# Function to get recipe from Edamam API
def get_recipe(query, tags):
    params = {
        'q': query,
        'app_id': EDAMAM_APP_ID,
        'app_key': EDAMAM_API_KEY,
        'health': tags
    }
    # print(params)
    response = requests.get(EDAMAM_API_ENDPOINT, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("hits"):
            return random.choice(data["hits"])["recipe"]
    return None

# Fetch the latest meal suggestions
latest_suggestions = results_collection.find_one(sort=[("timestamp", -1)])
# print(latest_suggestions)
user_summaries = list(user_summary_collection.find({}))

if latest_suggestions and user_summaries:
    combined_user_summary = {summary['username']: summary for summary in user_summaries}
    applicable_tags = determine_applicable_tags(combined_user_summary)
    timestamp = datetime.now()  # Set a single timestamp for all operations in this batch

    recipe_details = []  # To track if any recipes were processed

    for suggestion in latest_suggestions.get('suggestions', []):
        # Processing the meal
        meal_recipe = get_recipe(suggestion['meal'], applicable_tags)
        if meal_recipe:
            recipes_collection.insert_one({
                'username': 'admin',
                'timestamp': timestamp,
                'recipe': meal_recipe,
                'type': 'meal',
                'name': suggestion['meal']
            })
            recipe_details.append(suggestion['meal'])

        # Processing the sides
        for side in suggestion.get('sides', []):
            side_recipe = get_recipe(side, applicable_tags)
            if side_recipe:
                recipes_collection.insert_one({
                    'username': 'admin',
                    'timestamp': timestamp,
                    'recipe': side_recipe,
                    'type': 'side',
                    'name': side
                })
                recipe_details.append(side)

    # # Send a notification only if any recipes were processed
    # if recipe_details:
    #     formatted_message = fetch_and_format_recipes('admin', timestamp)
    #     send_teams_notification(formatted_message, webhook_url)


# Close MongoDB connection
client.close()

print("Process Complete.")