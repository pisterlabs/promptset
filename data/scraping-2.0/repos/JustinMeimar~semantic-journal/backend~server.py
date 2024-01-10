from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import json
import os
import openai

from quantify_goals import *
from goal_creation import generate_metric

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


app = Flask(__name__)
CORS(app, resources={r"/*" : {"origins":"http://localhost:3000"}})

@app.route('/')
def index():
    return "hello"

def create_goal(goal, metrics):

    request_data = request.get_json()
    new_goal = {
        "goal": goal,
        "metrics": metrics, 
        "journals": []
    }

    print("out")
    goals_file_path = os.path.join(data_dir, 'goal.json')
    with open(goals_file_path, 'w') as file:
        json.dump(new_goal, file, indent=4)

@app.route('/gen_metrics', methods=['POST'])
def gen_metrics():

    request_data = request.get_json()

    if 'goal' not in request_data:
        return jsonify({"error": "Invalid request data"}), 400

    goal = request_data['goal']
    metrics = {f'metric-{i+1}': mock_metric
                    for i, mock_metric in enumerate(generate_metric(goal))}
    
    create_goal(goal, metrics)

    return jsonify(metrics)

@app.route('/add_journal', methods=['POST'])
def add_journal():
    request_data = request.get_json()

    if 'date' not in request_data or 'content' not in request_data:
        print(f"missing fields in request: {request_data}")
        return jsonify({"error": "Invalid request data"}), 400

    goals_file_path = os.path.join(data_dir, 'goal.json')
    
    if os.path.isfile(goals_file_path):
        with open(goals_file_path, 'r') as file:
            goal_data = json.load(file)
    else:
        return jsonify({"error": "Goal not found"}), 404

    # TODO: proprely implement the GPT stuff  
    # call functions to get numbers from prompts
    print("goals are", goal_data['metrics'])
    convos = init_chat(goal_data['metrics'])
    nums = get_nums(convos, request_data['content'])

    # nums = [7, 8, 9]
    existing_journal = next((item for item in goal_data['journals'] if item['date'] == request_data['date']), None)

    if existing_journal:
        existing_journal['content'] = request_data['content']
        existing_journal['quantities'] = nums
    else:
        new_journal_entry = {
            "date": request_data['date'],
            "content": request_data['content'],
            "quantities": nums
        }
        goal_data['journals'].append(new_journal_entry)

    with open(goals_file_path, 'w') as file:
        json.dump(goal_data, file, indent=4)

    return jsonify({"message": "Journal updated successfully" if existing_journal else "Journal added successfully"})

@app.route('/get_journals', methods=['GET'])
def get_journals():
    goals_file_path = os.path.join(data_dir, 'goal.json')

    if os.path.isfile(goals_file_path):
        with open(goals_file_path, 'r') as file:
            goal_data = json.load(file)
        return jsonify(goal_data.get('journals', []))
    else:
        return jsonify({"error": "Goal not found"}), 404

@app.route('/get_goal', methods=['GET'])
def get_goal():
    goals_file_path = os.path.join(data_dir, 'goal.json')

    if os.path.isfile(goals_file_path):
        with open(goals_file_path, 'r') as file:
            goal_data = json.load(file)
        return jsonify(goal_data)
    else:
        return jsonify({"error": "Goal not found"}), 404

def gpt_response(user_message, goal_data):
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    try:
        CLIENT = OpenAI()

        messages = [
            {"role": "system", "content": "You are Samantha, the semantic journal assistant.\
             The following is a journal for tracking progress towards a goal.\
             please help the user with achieving their goal and evaluating their metrics"},
            {"role": "system", "content": json.dumps(goal_data, indent=4)},
            {"role": "user", "content": user_message}
        ]
        
        response = CLIENT.chat.completions.create(
            model='gpt-4',
            messages=messages,
        )

        print(response)
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "Sorry, I couldn't process that."

    
@app.route('/send_message', methods=['POST'])
def send_message():
    request_data = request.get_json()

    if 'message' not in request_data or request_data['message'] is None:
        return jsonify({"error": "Message is required"}), 400

    user_message = request_data["message"]

    try:
        with open("data/goal.json", "r") as file:
            goal_data = json.load(file)
        gpt_response_text = gpt_response(user_message, goal_data)
        
        return jsonify({"response": gpt_response_text})

    except FileNotFoundError:
        return jsonify({"error": "Goal file not found"}), 404

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in goal file"}), 500

if __name__ == '__main__':
    app.run(debug=True)
