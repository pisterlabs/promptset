'''
1. The function accepts a list of prompts as its parameter.
2. For each prompt, we record the start time and then use openai.Completion.create to generate a response from OpenAI. We limit the response to 100 tokens, but you can adjust this limit based on your needs.
3. We then record the end time and calculate the time taken for the generation.
4. We store the response text and the time taken in a tuple, and append this tuple to our list of responses.
5. Finally, we return the list of responses.
'''

import openai
import secret
import time

openai.api_key = secret.api_key

def Generate(prompts):
    responses = []
    
    for prompt in prompts:
        start_time = time.time()
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        response_text = response.choices[0].message['content'].strip()
        
        responses.append((response_text, time_taken))
    
    return responses

# List of prompts
prompts = [
    "Who won the world series in 2020?",
    "What's the capital of France?",
    "Who wrote Pride and Prejudice?"
]

# Generate responses
responses = Generate(prompts)

# Print responses and time taken
for i, (response, time_taken) in enumerate(responses):
    print(f"Prompt: {prompts[i]}\nResponse: {response}\nTime Taken: {time_taken} seconds\n")
