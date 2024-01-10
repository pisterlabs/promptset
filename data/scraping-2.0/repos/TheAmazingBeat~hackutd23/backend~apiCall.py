import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
is_Residential = True
num_Bathrooms = 3
num_Bedrooms = 4
sq_Feet = 2500
location = "Garland, Tx"
age = 24
prompt = ''

if(is_Residential):
    prompt = "Make a professional value proposition given these parameters: number of bathrooms: " + str(num_Bathrooms) + ", num of bedrooms: " + str(num_Bedrooms) + ", total square feet: " + str(sq_Feet) + ", located in: " + location + ", age: " + str(age)
else:
    prompt = prompt = "Make a detailed value proposition given these parameters: commercial building, total square feet: " + str(sq_Feet) + ", location: " + str(location) + ", age: " + str(age)

print(prompt)

def getAPICall():

    completion = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt = prompt,
        max_tokens = 1000,
        temperature = 1
    )
    for choice in completion.choices:
        text = choice.text.replace('\n', '')
        print(choice.text)


getAPICall()
