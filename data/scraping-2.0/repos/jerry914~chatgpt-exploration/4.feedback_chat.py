import openai
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv('OPENAI_API_KEY')

class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are the instructor who teaching Ruby, and you gave your student the homework 'FizzBuzz'. Now you have to correct the code your student submits, the ultimate goal is to make the student learn how to write correct function code and good refactoring. Please note that you should not give the student the answer, but just give the instructions."}]
        self.turns = 0

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        if role == 'user':
            self.turns += 1

    def get_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        return response['choices'][0]['message']['content']

    def is_conversation_over(self):
        return self.turns >= 20

# Initialize a conversation for each student
conversations = {student_id: Conversation() for student_id in range(100)}

# Example usage:
student_id = 0  # or whichever student is sending a message
while not conversations[student_id].is_conversation_over():
    # Get user's message from input
    message = input("You: ")
    conversations[student_id].add_message("user", message)  
    response = conversations[student_id].get_response()
    print(response)