import pandas as pd
from openai import OpenAI
import time
import os
import concurrent.futures


# Set your OpenAI API key
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'], 
)

# Function to create the prompt
def create_prompt(code):
    return f"""I have a section of decompiled C code with function and variable names that are difficult to understand. I need help in renaming these functions and variables to be more human-readable and sensible. Here is the code snippet: \n{code}\n Please provide a list of all original variable and function names and their proposed new names in two JSON formats like {{\\"variables\\": [ {{\\"originalName\\": \\"\\", \\"newName\\": \\"\\"}}]}} and {{\\"functions\\": [ {{\\"originalName\\": \\"\\", \\"newName\\": \\"\\"}}]}}. I only need the list of variable names in the JSON, nothing else in the response, even a single word."""

# Function to send the prompt to GPT-4 and get the response
def get_gpt4_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a reverse engineer helper."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except:
        print(f"An error occurred: {e}")
        return "Timeout or error occurred." 

# Read the CSV file
df = pd.read_csv('p2_s1&2.csv')

# Create a new column for responses
df['fv_naming'] = ''

counter = 0
# Loop through each row
for index, row in df.iterrows():
    prompt = create_prompt(row['decompiled'])

    #response = get_gpt4_response(prompt)
    # Using concurrent.futures to implement a timeout
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_gpt4_response, prompt)
        try:
            response = future.result(timeout=25)  # Timeout in seconds
        except concurrent.futures.TimeoutError:
            response = "Timeout occurred."
    
    df.at[index, 'fv_naming'] = response
    print(f"{(counter / 10) * 100}%")
    counter+=1

# Write the updated dataframe to a new CSV file
df.to_csv('p1_s3.csv', index=False)