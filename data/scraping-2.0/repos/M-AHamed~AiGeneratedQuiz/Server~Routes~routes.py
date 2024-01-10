from flask import Blueprint, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
# load the .env file containing the API key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
quiz_routes = Blueprint('quiz_routes', __name__)
# function to generate the quiz takes 
# top of quiz and number of questions
def create_quiz(topic, num_questions):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106", # this model returns a json response
            response_format={"type": "json_object"}, # to return json response
            # message includes teh topic and number of questions
            messages=[
                {"role": "system", "content": "You are a quiz creator designed to output JSON. Given a topic and a number of questions, you will return a quiz. Multiple choice with 4 choices for each question, and one correct answer marked with 'true'. Return the quiz questions in this format: { 'question': 'What is the capital of France?', 'choices': {'A': 'Paris', 'B': 'London', 'C': 'Berlin', 'D': 'Madrid'}, 'answer': {'A': true} }."},
                {"role": "user", "content": f"Topic: {topic}, number of questions: {num_questions}"}
            ]
        )
        # quiz text is extracted from the response
        quiz_text = response.choices[0].message.content
        print("Quiz generated successfully ----------------------------------------")
        print(quiz_text)
        return quiz_text
    # catch excaptions from the API
    except Exception as e:
        return str(e)
    
# Quiz generation POST route
# at /generate_quiz
@quiz_routes.route('/generate_quiz', methods=['POST'])
def generate_quiz_route():
    json_data = request.get_json()
    quiz_topic = json_data.get('quiz_topic')
    number_of_questions = json_data.get('number_of_questions')
# check if the topic and number of questions are valid
    if not quiz_topic or not isinstance(quiz_topic, str):
        return jsonify({'error': 'Quiz topic is required and must be a string'}), 400
    if not number_of_questions or not isinstance(number_of_questions, int):
        return jsonify({'error': 'Number of questions is required and must be an integer'}), 400
# call the create_quiz function
    quiz_response = create_quiz(quiz_topic, number_of_questions)
# check if the response is a string
    if isinstance(quiz_response, str):
        try:
            # Parse the string response to JSON
            quiz_json = json.loads(quiz_response)
            # return the json to the client-side, frontend
            return jsonify(quiz_json)
        except json.JSONDecodeError:
            # Handle the case where the response is not a valid JSON string
            return jsonify({'error': 'Failed to parse quiz response'}), 500
    else:
        return jsonify({'error': quiz_response}), 500

# Quiz generation GET route for testing
@quiz_routes.route('/generate_quiz_get', methods=['GET'])
def generate_quiz_get_route():
    return "Quiz Generation GET route"
