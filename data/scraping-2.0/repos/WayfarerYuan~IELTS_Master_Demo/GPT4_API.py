import time
import threading
import openai
import sys

# Set the API key
openai.api_key = ""

# Initialize the prompt
# topic = "More and more business meetings and business training have taken place on the internet. Do the advantages of this development outweigh the disadvantages?"
prompt = [
    {
        "role": "system",
        "content": "Please evaluate the following IELTS essay and provide an overall score as well as scores based on the four criteria: Task Achievement, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy. Provide an overall comment, scores, reasoning, and suggestions for improvement for each criterion. \n\nReturn all the results in Chinese and in JSON format. \n\nThe essay is as follows:\n\n"
    },
]

# Function to print dots
def print_dots(stop_event):
    while not stop_event.is_set():
        sys.stdout.write('.')
        time.sleep(0.5)
        sys.stdout.flush()

# read the prompt from writting.txt under the same directory as this .py file
with open("writing0413.txt") as f:
    user_input = f.read()

# replace the /n or /t with a space in the user input
user_input = user_input.replace("\n", " ")
user_input = user_input.replace("\t", " ")

# Add the user input to the conversation history
prompt.append({"role": "user", "content": user_input})
print(f'prompt: {prompt}')

# Start the dots printing in a separate thread
stop_event = threading.Event()
dots_thread = threading.Thread(target=print_dots, args=(stop_event,))
dots_thread.start()

# Generate a response
response = openai.ChatCompletion.create(
    # model = "gpt-3.5-turbo",
    # model="gpt-3.5-turbo-16k",
    model="gpt-4",
    messages=prompt,
    max_tokens=7000,
    temperature=0.1,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
)

# Stop the dots printing
stop_event.set()
dots_thread.join()

# Extract the assistant's message and print it
assistant_message = response['choices'][0]['message']['content']
print(f"\nGPT: {assistant_message}")

# Add the assistant's response to the conversation history
prompt.append({"role": "assistant", "content": assistant_message})
