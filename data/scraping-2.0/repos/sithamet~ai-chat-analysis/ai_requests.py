import openai
import os
from dotenv import load_dotenv
from datetime import datetime
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Load the .env file
load_dotenv()

# Get the OpenAI API key from the .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure the OpenAI API key
openai.api_key = openai_api_key

# Initialize list to accumulate token usage and a Lock for thread-safety
token_data = []
token_lock = Lock()

def make_ai_request(system_prompt, user_input, user_prompt="", model="gpt-3.5-turbo-16k", max_tokens=200):
    """
    Make a request to the OpenAI API with the given prompt and model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.5
    )

    assistant_response = response['choices'][0]['message']['content'].strip()

    input_tokens = sum([len(message['content']) for message in messages])
    output_tokens = len(assistant_response)

    # Thread-safe token accumulation
    with token_lock:
        token_data.append((input_tokens, output_tokens))

    return assistant_response

def make_parallel_requests(prompt_data, num_threads=10):
    """
    Makes parallel API requests based on the given prompt data.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(make_ai_request, **data): data for data in prompt_data}
        results = {}
        for future in futures.as_completed(futures):
            data = futures[future]
            try:
                results[data] = future.result()
            except Exception as e:
                print(f"Failed to get a result for prompt {data}: {e}")
        return results

# ... (Rest of the code remains the same)



def log_token_usage():
    """
    Log the accumulated token usage to a file.
    """
    # Create a file with a timestamp and session name
    timestamp = datetime.now().strftime('%d-%m-%y-%H-%M-%S')
    session_name = f"session_{timestamp}"
    file_name = f"{session_name}_tokens.txt"

    # Initialize variables to hold the sum of input and output tokens
    total_input_tokens = 0
    total_output_tokens = 0

    # Write accumulated token data to the file
    with open(file_name, "w") as file:
        for i, (input_tokens, output_tokens) in enumerate(token_data):
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens


        #define input cost, which is 	$0.003 / 1K tokens
        input_cost = total_input_tokens * 0.003 / 1000

        #define output cost, which is 	$0.004 / 1K tokens
        output_cost = total_output_tokens * 0.004 / 1000

        # Write the sum of input and output tokens for the entire session to the file
        file.write(f"Total Input Tokens for Session: {total_input_tokens} for ${input_cost} \n")
        file.write(f"Total Output Tokens for Session: {total_output_tokens} for ${output_cost}\n")
        file.write(f"Total Tokens for Session: {total_input_tokens + total_output_tokens} for ${input_cost + output_cost} \n")

        # Print the sum of input and output tokens for the entire session to the console
        print(f"Total Input Tokens for Session: {total_input_tokens} for ${input_cost} \n")
        print(f"Total Output Tokens for Session: {total_output_tokens} for ${output_cost}\n")
        print(f"Total Tokens for Session: {total_input_tokens + total_output_tokens} for ${input_cost + output_cost} \n")

