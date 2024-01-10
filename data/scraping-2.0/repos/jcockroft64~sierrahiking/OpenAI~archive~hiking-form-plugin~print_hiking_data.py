#!/usr/bin/python3
import sys
#import subprocess
#subprocess.run(['/usr/bin/python3', '-m', 'pip', 'install', 'openai', '--user'])
import openai

#get user input
age = sys.argv[1]
weight = sys.argv[2]
hike_length = sys.argv[3]

print("Age: " + age)
print("Weight: " + weight)
print("Length of Hike: " + hike_length + " days")

#ask ChatGPT some prompts concerning the backpacking list, among other things.

# Set up OpenAI API key
openai.api_key = "Valid OpenAI Key"

# Define function to generate hiking list
def generate_hiking_list(age, weight, hike_length):
    prompt = f"Generate a hiking list for a {age}-year-old, {weight}-pound person going on a {hike_length}-day hike in the Sierra Nevadas."
    prompt += "The list should be displayed in 2 columns, with the left column being the item and the right column being its weight in pounds."
    completion = openai.ChatCompletion.create( # 1. Change the function Completion to ChatCompletion
        model = 'gpt-3.5-turbo',
        messages = [ # 2. Change the prompt parameter to the messages parameter
            {'role': 'user', 'content': prompt}
        ],
        temperature = 0  
    )
    return completion['choices'][0]['message']['content']

hiking_list = generate_hiking_list(age, weight, hike_length)
print("Backpacking List:")
print(hiking_list)