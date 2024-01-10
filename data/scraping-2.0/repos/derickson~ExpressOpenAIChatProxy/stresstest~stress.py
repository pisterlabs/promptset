import argparse
import concurrent.futures
import threading
import time

import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "dave"
openai.api_base = "http://localhost:3000/v1"

counter_lock = threading.Lock()
total_calls = 0
error_calls = 0
sleetime = 1



# The function you want to stress test
def function_to_stress_test():
    global total_calls
    global error_calls
    global sleetime


    # Replace this with the function you want to test
    # For example, it could be an API call or any computationally intensive task
    time.sleep(sleetime)

    chat_completion = None
    try:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI asistant that helps write the plot to science fiction movies by playing the role of a TV news anchor in an interview on a science fiction show called 'The End of the World'. "},
                {"role": "user", "content": "hello what's in the news today?"},
                {"role": "assistant", "content": "in top news stories a giant asteroid is hurtling towards the earth"},
                {"role": "user", "content": "what should we do about that?"},
                ]
            )
        # print(chat_completions["choices"][0]["message"]["content"])
        with counter_lock:
            total_calls += 1
    except openai.error.OpenAIError as e:
        # Handle exceptions here, if any
        with counter_lock:
            error_calls += 1
        # print("Error occurred:", e)

    # print("Function call completed")

def stress_test_function(num_threads = 2, num_iterations = 50):
    # Number of threads to run in parallel
    # num_threads = 2

    # Number of times each thread should call the function
    # num_iterations = 50

    # Create a ThreadPoolExecutor with the desired number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the function to the executor multiple times
        futures = [executor.submit(function_to_stress_test) for _ in range(num_iterations * num_threads)]

        # Wait for all the futures to complete
        concurrent.futures.wait(futures)
    print(f"{args.num_threads}, {args.num_iterations}, {total_calls}, {error_calls}")

# if __name__ == "__main__":
#     stress_test_function(2, 10)
#     print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test a function by calling it in parallel on multiple threads.")
    parser.add_argument("sleep_time", type=float, help="seconds to sleep between each call on a thread")
    parser.add_argument("num_threads", type=int, help="Number of threads to run in parallel.")
    parser.add_argument("num_iterations", type=int, help="Number of times each thread should call the function.")
    args = parser.parse_args()

    sleetime = args.sleep_time
    # print(f"NUM_THREADS, NUM_QUERIES_PER_THREAD,SUCCESS,ERRORS")
    stress_test_function(args.num_threads, args.num_iterations)
    