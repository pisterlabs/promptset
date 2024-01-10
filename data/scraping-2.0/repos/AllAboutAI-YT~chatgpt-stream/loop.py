import openai
from openai import OpenAI
import os
import time

# ANSI escape code for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def open_file(filepath):
    """Open and read the content of a file."""
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
# Set the OpenAI API key
api_key = open_file('openaiapikey.txt')

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)        
   

def chatgpt_streamed(user_input):
    """
    Function to send a query to OpenAI's GPT-3.5-Turbo model, stream the response, and print each full line in yellow color.

    :param user_input: The query string from the user.
    :return: The complete response from the OpenAI GPT model after all chunks have been received.
    """
    # Send the query to the OpenAI API with streaming enabled
    streamed_completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": user_input}
        ],
        stream=True  # Enable streaming
    )

    # Initialize variables to hold the streamed response and the current line buffer
    full_response = ""
    line_buffer = ""

    # Iterate over the streamed completion chunks
    for chunk in streamed_completion:
        # Extract the delta content from each chunk
        delta_content = chunk.choices[0].delta.content

        # If delta content is not None, process it
        if delta_content is not None:
            # Add the delta content to the line buffer
            line_buffer += delta_content

            # If a newline character is found, print the line in yellow and clear the buffer
            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                for line in lines[:-1]:  # Print all but the last line (which might be incomplete)
                    print(NEON_GREEN + line + RESET_COLOR)
                    full_response += line + '\n'
                line_buffer = lines[-1]  # Keep the last line in the buffer

    # Print any remaining content in the buffer in yellow
    if line_buffer:
        print(NEON_GREEN + line_buffer + RESET_COLOR)
        full_response += line_buffer

    # Return the assembled full response
    return full_response    

def improve_solution(problem, iterations=5):
    """
    Improve a solution iteratively by feeding it back into the API.
    """
    solution = chatgpt_streamed(problem)
    for _ in range(iterations):
        improvement_prompt = f"Suggested Soulution: {solution} Based on this problem: {problem} make improvments to the Code Above:"
        solution = chatgpt_streamed(improvement_prompt)
    return solution

# Read the problem statement
problem = open_file('problem2.txt')

# Improve the solution iteratively
improved_solution = improve_solution(problem, iterations=5)  # Change iterations as needed
print("\nFinal Improved Solution:\n", improved_solution)
