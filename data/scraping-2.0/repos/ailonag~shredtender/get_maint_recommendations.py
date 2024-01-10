import openai
import mysql.connector
import datetime
import requests
import json

# Set your OpenAI API key
api_key = "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"
def generate_text(prompt, max_tokens):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

# Connect to the MariaDB
db = mysql.connector.connect(
    host="10.0.0.66",
    user="tunerapp",
    password="sN*mnswB7SHipiRB",
    database="suspension_tuner",
    port=3307,
)

cursor = db.cursor()

#

# Define the prompt and other parameters for the model
prompt = "Based on the bike stats: Total Hours: 100, Total Miles: 2000, Average Watts: 150, Average Suffer Score: 50, Total Elevation Gain (ft): 1000, Average Max Speed: 20, Normal Rides: 50, Workouts: 10, Races: 5, what maintenance recommendations do you have?"

# Query the GPT-4 model
response = openai.Completion.create(
    engine="text-davinci-003", # You may need to adjust the engine based on availability
    prompt=prompt,
    max_tokens=150
)

# Extract the recommendation from the response
recommendation = response.choices[0].text

# Print or send the recommendation back to the client
print(recommendation)   