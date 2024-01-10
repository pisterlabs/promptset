from flask import Flask , jsonify, Blueprint, request
from flask_cors import CORS, cross_origin
import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
os.environ['OPENAI_API'] = os.getenv('OPEN_AI_API')


app = Flask(__name__)
get_Modules = Blueprint('get_Modules', __name__)


# ==== Helper Functions ====
def getListofModules(response):
    # Convert the response to a list of modules
    modules = []
    for i in range(len(response['choices'])):
        modules.append(response['choices'][i]['text'])
    return modules

def getModulesfromCourse(course_name_prompt):

    openai.api_key = os.environ['OPENAI_API']

    # Call the OpenAI API to generate a response

    response = openai.ChatCompletion.create(

        model="gpt-3.5-turbo",

        messages=[{

            "role": "system",

            "content": "You are a fun yet knowledgable assistant."

        }, {

            "role": "user",

            "content": course_name_prompt

        }],

        temperature=0.6,

        max_tokens=1000)
    
    res = json.loads(response.choices[0].message.content)
    

    # Return the generated modules
    return res



# CORS(app)

@get_Modules.route('/get_Modules',methods=['POST'])
def get_data():

    course_name = request.get_json()['userInput']
    course_name_prompt = "What are the modules for " + course_name + "?.Give your response as a JSON like this: \`{ topics: Array<string> \`}" + "Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    # Your route logic here
    return getModulesfromCourse(course_name_prompt)
