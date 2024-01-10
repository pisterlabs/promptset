from flask import Flask , jsonify, Blueprint, request
from flask_cors import CORS, cross_origin
import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
os.environ['OPENAI_API'] = os.getenv('OPEN_AI_API')


app = Flask(__name__)
get_Quiz = Blueprint('get_Quiz', __name__)


# ==== Helper Functions ====
def getListofModules(response):
    # Convert the response to a list of modules
    modules = []
    for i in range(len(response['choices'])):
        modules.append(response['choices'][i]['text'])
    return modules
def getQuizQuestions(prompt):

    openai.api_key = os.environ['OPENAI_API']
    # Call the OpenAI API to generate a response

    response = openai.ChatCompletion.create(

        model="gpt-3.5-turbo",

        messages=[{

            "role": "system",

            "content": "You are a fun yet knowledgable assistant."

        }, {

            "role": "user",

            "content": prompt

        }],

        temperature=0.6,

        max_tokens=1000)
    
    res = json.loads(response.choices[0].message.content)
    

    # Return the generated modules
    return res



# CORS(app)

@get_Quiz.route('/get_Quiz',methods=['POST'])
def get_data():

    article = request.get_json()['userInput']
    quiz_prompt = "Give me 5 MCQ questions for student who read this : " + article + ".Give your response as a JSON like this: {'quiz' : Array <{ 'question' : string , 'options' : Array<string>, 'answer' : string }>}" + "Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    # Your route logic here
    return getQuizQuestions(quiz_prompt)
