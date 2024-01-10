import os
import openai
import time
import multiprocessing

api_key = "YOUR_API_KEY_HERE"

# Initialize the OpenAI API client with your API key
openai.api_key = api_key

def api_call(result_queue, prompt, input_data, max_tokens=1400):
    try:
        # Call the OpenAI API with the provided step and prompt
        message_log = [
            {
                "role": "user",
                "content": f"{prompt}\n{input_data}"
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_log,
            max_tokens=max_tokens,
            stop=None,
            temperature=0.7,
        )

        assistant_reply = response["choices"][0]["message"]["content"]
        result_queue.put(assistant_reply)

    except Exception as e:
        return e

def timeout_check(result_queue, timeout_seconds):
    time.sleep(timeout_seconds)
    result_queue.put(timeout_seconds)

import multiprocessing

def gpt_api_call(prompt, input_data, max_tokens=1400, max_retries=3, timeout_seconds=30):
    """
    Calls the GPT API with the given prompt and input data, and returns the response.
    
    Args:
        prompt (str): The prompt to send to the GPT API.
        input_data (str): The input data to send to the GPT API.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1400.
        max_retries (int, optional): The maximum number of times to retry the API call in case of failure. Defaults to 3.
        timeout_seconds (int, optional): The maximum number of seconds to wait for the API response before timing out. Defaults to 30.
    
    Returns:
        str: The response from the GPT API.
    
    Raises:
        Exception: If the maximum number of API call attempts is reached without success.
    """
    terminated = False
    for i in range(max_retries):
        result_queue = multiprocessing.Queue()
        process_api_call = multiprocessing.Process(target=api_call, args=(result_queue, prompt, input_data, max_tokens))
        process_timeout_check = multiprocessing.Process(target=timeout_check, args=(result_queue, timeout_seconds,))

        process_api_call.start()
        process_timeout_check.start()

        while True:
            if not process_api_call.is_alive():
                # API replied before Timeout
                process_timeout_check.terminate()
                response = result_queue.get()
                terminated = True
                return response
            
            if not process_timeout_check.is_alive():
                # Timeout expired before API reply
                process_api_call.terminate()
                timeout = result_queue.get()
                if timeout is not str:
                    timeout = str(timeout)
                print("(!)" + timeout + "s timeout expired." + " Attempt " + str(i+1) + "/" + str(max_retries))
                break

        if not terminated and i == max_retries - 1:
            # MAX ATTEMPTS REACHED
            raise Exception("(!) Max API attempts reached. Stopping the program.")



def load_prompt(prompt_path):
    try:
        with open(prompt_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {prompt_path} was not found.")
        return None