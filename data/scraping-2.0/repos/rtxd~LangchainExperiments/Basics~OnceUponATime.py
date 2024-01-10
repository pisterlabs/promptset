from dotenv import load_dotenv
from langchain.llms import OpenAI
import os

# Warning this is using openAI package version <1.0.0

#Load the .env file
load_dotenv()

#Access the 'SECRET_KEY' variable
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=openai_api_key)

result = llm("Once upon a time", max_tokens=5)
print(result)