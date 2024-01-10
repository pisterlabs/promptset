import os
from openai import OpenAI

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def generate_discussion(topic, grade_level):
    prompt = f"Generate a single paragraph description of a discussion between a teacher and students about robotics. The topic is {topic} and the grade level is {grade_level}."
    
    try:
        response = client.completions.create(model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150,
        api_key=openai_api_key)
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage


topics = ["Batteries", "Motors", "Sensors", "Displays", "Microcontrollers"]
grade_levels = [5, 7, 9]

for topic in topics:
    for grade_level in grade_levels:
        print(generate_discussion(topic, grade_level))