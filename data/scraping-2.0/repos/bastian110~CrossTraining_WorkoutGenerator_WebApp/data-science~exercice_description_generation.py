import openai

import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')



# Load your dataset
df = pd.read_csv("/Users/bastianchuttarsing/Documents/Crosstraining_WebApp/data-science/dataset/dataset_exercices_crosstraining.csv")

# Define the prompt for the API to generate descriptions
prompt = (
    "Ignore all instructions before this one. You're"
    "Crossfit and cross-training coach. You have been coaching crossfitter and sportives for 20 years."
    "Your task is now to explain me in one sentence the principe of this exercice: {exercise_name}"
    
)

# Create a new column for the generated descriptions
df["short_description"] = ""

# Loop through each row in the dataset and generate a description
for index, row in df.iterrows():
    exercise_name = row["Name"]
    
    # Format the prompt with the exercise details
    input_text = prompt.format(
        exercise_name=exercise_name,
    )
    
    # Use the OpenAI GPT API to generate a description
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.1,
        seed=123
    )
    
    # Extract the generated description from the API response
    description = response.choices[0].text.strip()
    print(description)
    
    # Assign the generated description to the corresponding row in the dataframe
    df.at[index, "short_description"] = description

# Save the updated dataframe to a new file
df.to_csv("/Users/bastianchuttarsing/Documents/Crosstraining_WebApp/data-science/dataset/dataset_exercices_crosstraining_with_description.csv", index=False)