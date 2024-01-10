import openai
from datetime import datetime
import os

# Initialize folder for saving responses
if not os.path.exists('responses'):
    os.mkdir('responses')

# Define a function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Define a function to save content to a file
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

# Initialize OpenAI API key
api_key = open_file('openaiapikey2.txt')
openai.api_key = api_key

# Read the content of the files containing the chatbot's prompts
chatbot_prompt = open_file('sysprompt.txt')

# Initialize an empty list to store the conversations for the chatbot
conversation = []

def chatgpt(api_key, conversation, chatbot_prompt, solver, temperature=1.4, frequency_penalty=0.2, presence_penalty=0):
    conversation.append({"role": "user", "content": solver})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot_prompt}]
    messages_input.insert(0, prompt[0])
    
    completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)
    
    chat_response = completion['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": chat_response})
    
    return chat_response

# Number of loops / examples
num_loops = 2

for i in range(num_loops):
    problem = open_file('problems.txt')
    prob1 = chatgpt(api_key, conversation, chatbot_prompt, problem)
    solver = open_file('prompt1.txt').replace("<<SELLER>>", prob1)
    response = chatgpt(api_key, conversation, chatbot_prompt, solver)
    
    # Create a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"responses/response_{timestamp}.txt"
    
    # Combine the input prompt and response
    combined_content = f"Input Prompt:\n{prob1}\n\nResponse:\n{response}"
    
    # Save to a file
    save_file(filename, combined_content)
    print(f"Saved example {i+1} to {filename}")
    
    conversation.clear()