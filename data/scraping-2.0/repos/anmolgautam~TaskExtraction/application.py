import openai
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import gradio as gr
import tiktoken

load_dotenv()
current_timestamp = datetime.now()
formatted_time = current_timestamp.strftime('%H_%M_%S')
encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")

class Agent:
    def __init__(self, api_key, system_prompt, role):
        self.api_key = api_key
        self.conversation_log = ""
        self.system_prompt = system_prompt
        self.client = openai.Client(api_key=self.api_key)
        self.role = role

    def log_task(self, task):
        with open(f'./tasks/task_{formatted_time}.txt', 'a') as log:
            log.write(f"{task}\n")

    def get_response(self, prompt, system_prompt=None):
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            max_tokens=2000,
            temperature=0.5,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{self.system_prompt}"},
                {"role": "user", "content": f"{prompt}"},
                {"role": "assistant", "content": " "},
            ])
        out = dict(response)
        out = dict(out['choices'][0])
        out = dict(out['message'])
        out = out['content']

        print("Tokens used :", len(encoding.encode(self.system_prompt + ' ' + prompt)))

        return out

def add_to_conversation_log(self, user_input, response):
        self.conversation_log += "user: " + user_input + "\nassistant: " + response + "\n"

        if json.loads(response).get('Flag') == "True":
            self.log_task(json.loads(response)['Task_list'])

# Function that `gradio` will call when the user enters a prompt
def chat_with_agent(user_input, history):
    response = samantha.get_response(str(history) + user_input)
    #add_to_conversation_log(user_input, response)
    with open(f"./conversations/chat_{formatted_time}.txt",'w') as file:
        file.write(str(history))
    print(history)
    return response

# Initialize the agent
api_key = os.getenv("OPEN_AI_API")
with open('assistant_interaction_prompt.txt', 'r') as file:
    system_prompt = file.read()

with open('skill.json', 'r') as file:
    skills = json.load(file)

role = "SDR"
system_prompt = system_prompt.replace('{skills}', str(skills[role]['skills']))
system_prompt = system_prompt.replace('{role}', role)
samantha = Agent(api_key, system_prompt=system_prompt, role=role)

# Setup Gradio interface
iface = gr.ChatInterface(
    chat_with_agent,
    chatbot=gr.Chatbot(height = 600),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="SDR",
    description="Ask any question",
    theme="soft",
    examples=["Hi", "Find me leads", "Find me prospects"],
    #cache_examples=True,
    retry_btn=None,
    #undo_btn="Delete Previous",
    #clear_btn="Clear",
)

# Start the Gradio server
iface.launch()