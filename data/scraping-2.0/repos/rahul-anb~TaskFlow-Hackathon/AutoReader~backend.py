from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def process_message(message):
    # # Your processing logic goes here
    # # For example, let's just return the reversed message
    # return message[::-1]
    llm = OpenAI(temperature=0.6)
    
    # prompt = "you are a Task list creation aiding tool who will take in a input detail and give out an output in a specific format to aid in generating tasks and subtask for a project. Generated tasks should be very project specific and include specialized tasks in them. Create only 4 to 6 main headings unless specifically asked to. Create a task list for inventory management system"
    name = llm(message)
    
    return name

@app.route('/process', methods=['POST'])
def process():
    try:
        print("asd")
        data = request.get_json()
        message = data['prompt']
        print("check"+ message)
        processed_message = process_message(message)

        # You can add more data to the response if needed
        response_data = {'processed_message': processed_message}

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
