import openai
import os
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures
import time

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_response(prompt, temperature, top_p):
    system_message = "You are a helpful assistant."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        top_p=top_p,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content']

# Define a wrapper function that includes a timeout
def get_response_with_timeout(prompt, temperature, top_p, timeout=2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_response, prompt, temperature, top_p) 
        try:
            return future.result(timeout=timeout)  # Enforce the 3 seconds timeout
        except concurrent.futures.TimeoutError:
            return f"No response from server"

prompts = ["Write a tagline for an ice cream shop."]
# temperatures = [0.3, 1.7]  # Example temperature values
# top_ps = [0.25, 0.35, 0.45, 0.85, 0.95 ]       # Example top_p values
temperatures = [1.7]
top_ps = [ 0.4 ] 
num_requests = 8  # Number of requests per combination

csv_file_path = 'generated_responses_2.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['prompt', 'temperature', 'top_p', 'response']
    writer = pd.DataFrame(columns=fieldnames)

    # Write the header
    writer.to_csv(csvfile, index=False)

    # Generate responses
    for prompt in prompts:
        for temperature in temperatures:
            for top_p in top_ps:
                for _ in range(num_requests):
                    response = get_response_with_timeout(prompt, temperature, top_p)
                    print(response)
                    writer = pd.DataFrame([{
                            'prompt': prompt,
                            'temperature': temperature,
                            'top_p': top_p,
                            'response': response
                        }])
                    writer.to_csv(csvfile, header=False, index=False)

csv_file_path