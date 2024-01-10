import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


function_descriptions = [
    {
        "name": "extract_email",
        "description": "categroise and extract key info from an email, such as the name of the sender, a summary of the email, etc.",
        "parameters":{
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "the name of the sender of the email"
                },
                "category": {
                    "type": "string",
                    "description": "the category of the email, such as personal, client needs, promtional, advertisment, spam, coupon, etc."
                    },
                "summary": {
                    "type": "string",
                    "description": "a brief summary of the email in 15 words or less"
                },
                "priority": {
                    "type": "string",
                    "description": "Try to give a priority score to this email based on how important the information is for business, from 0 to 10; 10 most important. Factors that determine importance is urgency, relevance, and the sender of the email. Unimportance factors include promotional deals, spam, etc."
                }
            }
        },
        "required": ["name", "summary", "rating"],
    }
]

def extract_email(email_body):
    
    prompt = f"Please extract key information from this email: {email_body} "
    message = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model = "gpt-4-0613",
        temperature = 0.1,
        messages=message,
        functions = function_descriptions,
        function_call = "auto"
    )
    
    return response