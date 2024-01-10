# app/routes.py
import sys
import os
import json
from flask import jsonify

sys.path.append(r"D:\Projects\Git\nexa_lift")

from dotenv import load_dotenv

load_dotenv(dotenv_path="./app/ai_module/.env")

from flask import request, render_template
from app import app
from app.ai_module.openai_integration import OpenAIIntegration
from app.ai_module.prompts_eng import olympic_main_prompt

api_key = os.getenv("OPENAI_API_KEY")
organization = os.getenv("OPENAI_ORG_ID")
openai_integrator = OpenAIIntegration(api_key, organization)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        workout_query = request.form.get('workout_query')
        if workout_query:
            #response = openai_integrator.get_response(olympic_main_prompt)

            response = potential_workout_plans
            try:
                response_json = json.loads(response.replace("{...}", "null"))
            except json.JSONDecodeError:
                return jsonify(error="Invalid response format"), 400

            return jsonify(response_json)  # use jsonify to return a json response

        else:
            response = "Please enter a workout query"
            jsonify(response)

    return render_template('index.html')


potential_workout_plans = """
{
"Week 1": {
"Day 1": {
"Exercises": [
{
"Exercise": "Snatch",
"Muscle Group": "Full Body",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Front Squat",
"Muscle Group": "Legs",
"Reps": 5,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-1-1"
},
{
"Exercise": "Bench Press",
"Muscle Group": "Chest",
"Reps": 8,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "High",
"Reasoning": "Develop push movements and full-body power",
"Workout Name": "Strength & Conditioning - Push Dominant"
},
"Day 2": {
"Exercises": [
{
"Exercise": "Clean",
"Muscle Group": "Full Body",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Sumo Deadlift",
"Muscle Group": "Back & Legs",
"Reps": 5,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Pull-Up",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-0-2"
},
{
"Exercise": "Barbell Row",
"Muscle Group": "Back",
"Reps": 8,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "Moderate",
"Reasoning": "Develop pull movements and stability",
"Workout Name": "Strength & Conditioning - Pull Dominant"
},
"Day 3": {
"Exercises": [
{
"Exercise": "Clean & Jerk",
"Muscle Group": "Full Body",
"Reps": 2,
"Rest": "2 minutes",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Back Squat",
"Muscle Group": "Legs",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 8,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "2-1-1"
},
{
"Exercise": "Burpees",
"Muscle Group": "Full Body",
"Reps": 30,
"Rest": "NA",
"Sets": 1,
"Tempo": "1-0-1"
}
],
"Intensity": "High",
"Reasoning": "Overall strength, power, and endurance",
"Workout Name": "Full Body & Conditioning"
}
},
"Week 2": {
"Day 1": {
"Exercises": [
{
"Exercise": "Snatch",
"Muscle Group": "Full Body",
"Reps": 2,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Front Squat",
"Muscle Group": "Legs",
"Reps": 4,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-1-1"
},
{
"Exercise": "Bench Press",
"Muscle Group": "Chest",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "High",
"Reasoning": "Develop explosive power and strength",
"Workout Name": "Strength & Power"
},
"Day 2": {
"Exercises": [
{
"Exercise": "Clean",
"Muscle Group": "Full Body",
"Reps": 2,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Sumo Deadlift",
"Muscle Group": "Back & Legs",
"Reps": 4,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Pull-Up",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-0-2"
},
{
"Exercise": "Barbell Row",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "Moderate",
"Reasoning": "Develop pull movements and stability",
"Workout Name": "Strength & Conditioning - Pull Dominant"
},
"Day 3": {
"Exercises": [
{
"Exercise": "Clean & Jerk",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "2 minutes",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Back Squat",
"Muscle Group": "Legs",
"Reps": 5,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "2-1-1"
},
{
"Exercise": "Burpees",
"Muscle Group": "Full Body",
"Reps": 30,
"Rest": "NA",
"Sets": 1,
"Tempo": "1-0-1"
}
],
"Intensity": "High",
"Reasoning": "Overall strength, power, and endurance",
"Workout Name": "Full Body & Conditioning"
}
},
"Week 3": {
"Day 1": {
"Exercises": [
{
"Exercise": "Snatch",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Front Squat",
"Muscle Group": "Legs",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-1-1"
},
{
"Exercise": "Bench Press",
"Muscle Group": "Chest",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "High",
"Reasoning": "Develop explosive power and strength",
"Workout Name": "Strength & Power"
},
"Day 2": {
"Exercises": [
{
"Exercise": "Clean",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Sumo Deadlift",
"Muscle Group": "Back & Legs",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Pull-Up",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-0-2"
},
{
"Exercise": "Barbell Row",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "Moderate",
"Reasoning": "Develop pull movements and stability",
"Workout Name": "Strength & Conditioning - Pull Dominant"
},
"Day 3": {
"Exercises": [
{
"Exercise": "Clean & Jerk",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "2 minutes",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Back Squat",
"Muscle Group": "Legs",
"Reps": 4,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 8,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "2-1-1"
},
{
"Exercise": "Burpees",
"Muscle Group": "Full Body",
"Reps": 30,
"Rest": "NA",
"Sets": 1,
"Tempo": "1-0-1"
}
],
"Intensity": "High",
"Reasoning": "Overall strength, power, and endurance",
"Workout Name": "Full Body & Conditioning"
}
},
"Week 4": {
"Day 1": {
"Exercises": [
{
"Exercise": "Snatch",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Front Squat",
"Muscle Group": "Legs",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-1-1"
},
{
"Exercise": "Bench Press",
"Muscle Group": "Chest",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "High",
"Reasoning": "Develop explosive power and strength",
"Workout Name": "Strength & Power"
},
"Day 2": {
"Exercises": [
{
"Exercise": "Clean",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "90 seconds",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Sumo Deadlift",
"Muscle Group": "Back & Legs",
"Reps": 3,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Pull-Up",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "2-0-2"
},
{
"Exercise": "Barbell Row",
"Muscle Group": "Back",
"Reps": 6,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "3-0-2"
}
],
"Intensity": "Moderate",
"Reasoning": "Develop pull movements and stability",
"Workout Name": "Strength & Conditioning - Pull Dominant"
},
"Day 3": {
"Exercises": [
{
"Exercise": "Clean & Jerk",
"Muscle Group": "Full Body",
"Reps": 1,
"Rest": "2 minutes",
"Sets": 5,
"Tempo": "X-2-X"
},
{
"Exercise": "Back Squat",
"Muscle Group": "Legs",
"Reps": 4,
"Rest": "90 seconds",
"Sets": 4,
"Tempo": "3-1-1"
},
{
"Exercise": "Push Press",
"Muscle Group": "Shoulders",
"Reps": 8,
"Rest": "60 seconds",
"Sets": 3,
"Tempo": "2-1-1"
},
{
"Exercise": "Burpees",
"Muscle Group": "Full Body",
"Reps": 30,
"Rest": "NA",
"Sets": 1,
"Tempo": "1-0-1"
}
],
"Intensity": "High",
"Reasoning": "Overall strength, power, and endurance",
"Workout Name": "Full Body & Conditioning"
}
}
}"""