import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
# print('LLM - PHONE CASE CONVERSATION CONTINUE')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")
if not mongodb_uri:
    raise ValueError("MONGODB_URI is not set in environment variables")

# Connect to MongoDB
mongo_client = MongoClient(mongodb_uri)
db = mongo_client.product_bot  # Replace with your actual database name
conversations = db.conversations  # Replace with your actual collection name

# Create an OpenAI client instance
client = OpenAI(api_key=openai_api_key)

# Retrieve user response and conversation ID from arguments
user_response = sys.argv[1]
conversation_id = sys.argv[2]  # Get the conversation ID passed from app.py

# Retrieve the existing conversation from MongoDB
existing_conversation = conversations.find_one({"_id": conversation_id})
if not existing_conversation:
    print(f"No conversation found with ID {conversation_id}")
    sys.exit(1)


# Append the new user response to the conversation messages
existing_messages = existing_conversation.get("messages", [])
updated_messages = existing_messages + [
    {"role": "user", "content": user_response}
]

# Make the API call using the client instance
try:
    chat_completion = client.chat.completions.create(
        messages=updated_messages,
        model="gpt-3.5-turbo"  # Specify the model
    )

    if chat_completion.choices:
        generated_text = chat_completion.choices[0].message.content.strip()
        print(generated_text)

        # Update the conversation record with the new message
        conversations.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": {"role": "assistant", "content": generated_text}}}
        )
    else:
        print("No response received from the API.")

except Exception as e:
    print(f"An error occurred: {e}")
