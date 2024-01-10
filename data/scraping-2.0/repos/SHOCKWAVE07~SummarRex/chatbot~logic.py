# chatbot.py

import openai
import gradio as gr
from chatbot.roles import roles, role_selected,selected_model
with open('API_KEY', 'r') as file:
    api_key = file.read().strip()  

openai.api_key=api_key

# Initialize messages with a system message
messages = []



def CustomChatGPT(user_input, selected_role):
    # Update the selected model based on the role
    selected_model = roles[selected_role]["model"]
    
    # Append the system message with role description
    system_message = f"You are a {selected_role}. {user_input}"
    messages.append({"role": "system", "content": system_message})

    # Generate a response using the selected model
    response = openai.ChatCompletion.create(
        model=selected_model,
        messages=messages,
    )
    
    # Get the assistant's reply
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    
    # Append the user's input and assistant's reply to messages
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    
    return ChatGPT_reply


