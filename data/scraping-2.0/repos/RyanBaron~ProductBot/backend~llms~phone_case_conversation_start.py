import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
# print('LLM - PHONE CASE CONVERSATION START')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')  # Ensure this is set in your .env file

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

# Retrieve topic and conversation ID from arguments
topic = sys.argv[1]
conversation_id = sys.argv[2]  # Get the conversation ID passed from app.py

# Define the messages for the assistant
messages = [
    {
        "role": "assistant",
        "content": "To write effective Midjourney prompts, follow these guidelines: Use Correct Grammar: Midjourney requires precise grammar for accurate image generation. Choose synonyms for concise expression, like 'exhausted' instead of 'very tired.' Be Specific: Detail your ideas clearly. For example, use 'tyreless cycle' instead of 'bicycle with no tires.' Use Simple Language: Opt for straightforward language, keeping prompts neither too short nor too long. Give Reference: Reference styles, eras, or artists to guide the AI. Include Text Inputs: Specify location, subject, style, lighting, and emotions for detailed images. Use Parameters: Utilize technical inputs like aspect ratio (--aspect or --ar), exclude objects (--no), chaos (--chaos), and tile (--tile) for specific image traits."
    },
    {
        "role": "system",
        "content": "Your initial job will be to gather additional information that will help you write an effective midjourney prompt for the topic submitted by the user. You will do this through a series of simple questions that can be presented to the user giving them a choice between 2 options. Your 2nd task will be to generate a concise and effective midjourney prompt on the topic at hand based on the information you gathered from the user and you knowledge of midjourney prompting and best practices. You will generate a prompt in the format of '/imagine prompt:[prompt text goes here] --ar 5:8' any time the user says 'show me the prompt', otherwise you will contine to gather information to improve the prompt."
    },
    {
        "role": "user",
        "content": topic
    }
]

# Insert initial conversation record in MongoDB
# Define the conversation record
conversation_record = {
    "_id": conversation_id,  # Use the passed conversation ID
    "topic": topic,
    "product": "phone_case",
    "messages": messages
}
conversation_id = conversations.insert_one(conversation_record).inserted_id

# Make the API call using the client instance
try:
    chat_completion = client.chat.completions.create(
        messages=messages,
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

        with open("assistant_response.txt", "w") as f:
            f.write(generated_text)
    else:
        print("No response received from the API.")

except Exception as e:
    print(f"An error occurred: {e}")
