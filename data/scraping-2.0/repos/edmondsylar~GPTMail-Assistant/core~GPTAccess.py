import openai
import os
import dotenv

# load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# get api key from environment variables
openai.api_key = api_key

def get_ai_models():
    modeles = openai.Model.list()
    return modeles


def generate_mail_response(msg):
    messages = [
        {"role": "system", "content": """
        You are Edmond Musiitwa (AI) an AI email assistant. Your task is to generate a professional and polite response to the emil of email threads provided.
        analayze the email or email thread selecting the sender from the details shared and address them the most appropriate.
        
        If it's a marketing email, the response email should be a summery of the email content highligting the key points and how this can benefit me (Edmond Musiitwa). Only these kinds of emails responses will be addressed to Edmond Musiitwa as the receiver and Sign as my Personal AI Assistant.:
        
         """},
        {"role": "user", "content": msg},
    ]
    
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


# text = input("Enter your message: ")
# response = generate_mail_response(text)
# print (response)