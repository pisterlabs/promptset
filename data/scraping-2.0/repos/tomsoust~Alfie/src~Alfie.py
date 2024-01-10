from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
from openai import OpenAI
print('openai version:')
print(openai.__version__)

load_dotenv()




openai.api_key = os.environ.get("OPENAI_API_KEY")





template = """ You are an expert gift giver. 
              A person has come to you to get ideas about what gifts they should give a recipient. 
              Your task is:
              Ask the person up to 12 questions to make an informed suggestion on what this person should give the recipient. 
              Each question should build off the answer of the previous questions.
              Only ask one question at a time.
              DO NOT REPEAT A QUESTION

              After you have sufficient information about the person and the reciptant you must recommend a gift for the person to give the recipient.

 """


# Initial messages
initial_messages = [
    {"role": "system", "content": template},
    {"role": "assistant", "content": "what type of relationship do you have with the recipient?"},
]

# Create OpenAI client
client = OpenAI(
    api_key=openai.api_key,
)

app = Flask(__name__)
CORS(app)


@app.route('http://localhost:/5001/api', methods=['POST'])
def chat_with_openai(messages):
    first_question = "what type of relationship do you have with the recipient?"
    user_input = input(first_question)
    while True:
        # Get AI's response
        # Call OpenAI API
        ai_response = chat_model(messages,user_input)
        messages.append({"role": "assistant", "content": ai_response})
        # Update messages for the next iteration
        user_input = input(f"{ai_response}: ")
        # return jsonify(answer=ai_response)
    

def chat_model(messages, user_input):
  messages.append({"role": "user", "content": user_input})
  chat_completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=messages
      )
  ai_response = chat_completion.choices[0].message.content
  return ai_response


# Start the conversation
chat_with_openai(initial_messages)

    

if __name__ == '__main__':
    app.run(port=5001)