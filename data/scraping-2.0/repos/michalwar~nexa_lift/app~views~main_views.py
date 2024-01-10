import sys

sys.path.append(r"D:\Projects\Git\nexa_lift")

from dotenv import load_dotenv

load_dotenv(dotenv_path="./app/ai_module/.env")

from flask import request, render_template
from app import app
from app.ai_module.openai_integration import OpenAIIntegration

api_key = os.getenv("OPENAI_API_KEY")
organization = os.getenv("OPENAI_ORG_ID")
openai_integrator = OpenAIIntegration(api_key, organization)

@app.route('/', methods=['GET', 'POST'])
def index():
    workout_response = None
    if request.method == 'POST':
        workout_query = request.form.get('workout_query')
        workout_response = openai_integrator.get_response(workout_query)

    return render_template('index.html', workout_response=workout_response)
