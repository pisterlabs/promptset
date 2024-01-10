# new_app.py

import openai
import os
from dotenv import load_dotenv
load_dotenv()
import time 
import sys
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# MongoDB URI
uri = os.getenv("MONGO_CLIENT")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Select the appropriate database and collections
db = client["question_feedback"]
user_responses_collection = db["userResponses"]
feedback_collection = db["feedback"]
followup_collection = db["followUp"]

def verify_answer(question, user_answer):
    latest_feedback = feedback_collection.find_one(sort=[('_id', pymongo.DESCENDING)])

    # Check if a document was found in the collection
    if latest_feedback is None:
        verify_answer_id_counter = 1  # Set a default value if no documents are found
    else:
        verify_answer_id_counter = latest_feedback["_id"] + 1

    print("Verifying answer...")
    
    prompt = f"{question}\nUser answer: {user_answer}\nIs the answer correct? (yes or no)(make it yes if it is approximately correct(For example if the user is unable to provide the exact number.)) \n"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10
    )
    feedback_data = {}  # Initialize an empty dictionary
    feedback = response.choices[0].text.strip().lower()
    
    if feedback == "yes":

        time.sleep(20)
        prompt2 = f"Question: {question}\nUser answer: {user_answer}\n Speak about it (within 100 words)"
        response2 = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt2,
            max_tokens=200
        )
        feedback2 = response2.choices[0].text.strip().lower()
        
        feedback_data = {
            "_id": verify_answer_id_counter,
            "question": question,
            "user_answer": user_answer,
            "is_correct": True,
            "Interviewer": "Correct answer! Well done!",
            "feedback": feedback2
        }

    elif feedback == "no":
        feedback_data = {
            "_id": verify_answer_id_counter,
            "is_correct": False,
            "feedback": "It seems your answer needs clarification. Could you elaborate?"
        }
        user_answer1 = sys.argv[2]
        verify_answer(question, user_answer1)

    elif any(keyword in user_answer for keyword in ["don't know", "not sure", "no idea"]):
        feedback_data = {
            "_id": verify_answer_id_counter+1,
            "question": question,
            "user_answer": user_answer,
            "is_correct": False,
            "feedback": "That's okay! Mistakes happen. Remember, every attempt is a step towards learning.",
            "follow_up_question": None
        }
      
    else:
        feedback_data = {
            "_id": verify_answer_id_counter,
            "is_correct": False,
            "feedback": "Sorry. Can you repeat?"
        }
        user_answer2 = sys.argv[2]
        verify_answer(question, user_answer2)

    # print("Feedback data:", feedback_data)
    # Insert the feedback data into the MongoDB collection
    feedback_collection.insert_one(feedback_data)  
    sys.stdout.flush()
    pretty_feedback = json.dumps(feedback_data, indent=4)
    print(pretty_feedback)
    print(feedback_data['is_correct'])
    return feedback_data['is_correct']

def generate_followup_question(previous_question, user_answer):
    print("\nGenerating follow-up question...")

    latest_feedback = followup_collection.find_one(sort=[('_id', pymongo.DESCENDING)])

    # Check if a document was found in the collection
    if latest_feedback is None:
        followup_id_counter = 1  # Set a default value if no documents are found
    else:
        followup_id_counter = latest_feedback["_id"] + 1

    time.sleep(20)
    prompt = f"Based on your previous response:\n\nQ: {previous_question}\nA: {user_answer}\n\nGenerate a follow-up question:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    generated_text = response.choices[0].text.strip()

    print("Generated Follow-up Question:", generated_text)  # Debugging print statement

    followup_data = {
        "_id": followup_id_counter,  # Use the incremented variable as the _id
        "question": previous_question,
        "user_answer": user_answer,
        "follow_up_question": generated_text
    }

    # Insert the follow-up data into the MongoDB collection
    followup_collection.insert_one(followup_data)

    return generated_text



def main():
    try:
        # Fetch the latest user response from the database
        latest_response = user_responses_collection.find_one({}, sort=[('_id', pymongo.DESCENDING)])
        if latest_response:
            current_question = latest_response['question']
            user_answer1 = latest_response.get('answer')  
            print("Current Question:", current_question)
            print("User Answer:", user_answer1)

            is_correct = verify_answer(current_question, user_answer1)
            time.sleep(20)
            
            try:
                while is_correct:
                    # Generate follow-up question
                    followup_question = generate_followup_question(current_question, user_answer1)
                    user_answer2 = latest_response.get('answer') 
                    print("Running the verify function")
                    time.sleep(20)
                    verify_answer(followup_question, user_answer2)
                    current_question = followup_question
            except KeyboardInterrupt:
                print("\nScript interrupted. Exiting gracefully.")
        else:
            print("No user responses found in the database.")

    except Exception as e:
        print("Error:", str(e))

    # Close the MongoDB client connection
    client.close()

if __name__ == "__main__":
    main()
