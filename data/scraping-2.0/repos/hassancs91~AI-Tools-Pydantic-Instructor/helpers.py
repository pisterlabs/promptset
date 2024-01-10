#pydantic example
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

import instructor


#Replace With Your Open AI Key
open_ai_client = OpenAI(
     api_key=openai_api_key,
)

instructor.patch(open_ai_client)

def structured_generator(openai_model,prompt,custom_moel):
    result : custom_moel = open_ai_client.chat.completions.create(
        model = openai_model, 
        response_model = custom_moel,
        messages= [{"role":"user","content" : f"{prompt}, output must be in json"}]
    )
    return result


