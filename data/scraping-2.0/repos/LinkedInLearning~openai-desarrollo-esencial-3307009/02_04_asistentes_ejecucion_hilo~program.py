import time
import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

assistant = client.beta.assistants.create(
    name = "OpenAI para desarrollo esencial",
    instructions = "Eres un asistente muy útil.",
    tools=[{"type":"retrieval"}, {"type":"code_interpreter"}],
    model="gpt-4-1106-preview"
)

thread = client.beta.threads.create()
print(thread)

def display_main_menu():
    print("Asistente")
    prompt = input("\nEscribe el texto: ")
    handle_main_menu_option(prompt)

def handle_main_menu_option(prompt):
    message = client.beta.threads.messages.create(thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)    
    still_running = True
    while still_running:
        run_latest = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        still_running = run_latest.status != "completed"
        if (still_running):
            time.sleep(2)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages.data[0].content)

while True:
    display_main_menu()