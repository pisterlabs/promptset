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

def handle_main_menu_option(prompt):
    message = client.beta.threads.messages.create(thread.id, role="user", content=prompt)
    
    print(prompt)

while True:
    display_main_menu()