# filename: chat_operations.py

import streamlit as st
import openai

def calculate_token_count(user_input):
    if user_input is None:
        user_input = ""
    return len(user_input.split())

def update_token_count(selections_data):
    """Calculates the token count based on selections and displays in sidebar."""
    total_token_count = calculate_token_count(selections_data)
    st.sidebar.markdown(f"Current Tokens: {total_token_count}")

def trim_chat_history(chat_history, max_length, prompt, character, selected_systems_data, discussion_text):
    # Calculate the token count for the fixed parts of the context
    fixed_length = calculate_token_count(prompt) + calculate_token_count(character['introduction']) + calculate_token_count(selected_systems_data) + calculate_token_count(discussion_text)

    # Calculate the total token count for the chat history
    chat_length = sum(calculate_token_count(message) for message, _ in chat_history)

    # Determine the total length
    total_length = fixed_length + chat_length

    # Start trimming from the third message
    index_to_trim = 2
    while total_length > max_length and index_to_trim < len(chat_history):
        message, _ = chat_history.pop(index_to_trim)
        total_length -= calculate_token_count(message)
    return chat_history

def generate_response(user_input, chat_history, model_name, character, selected_systems_data, discussion_text):
    prompt = character['prompt'] + selected_systems_data + discussion_text
    messages = [{"role": "system", "content": prompt}]
    messages.append({"role": "system", "content": character['introduction']})
    messages += [{"role": "user" if sender == "Human" else "assistant", "content": message} for message, sender in chat_history]
    messages.append({"role": "user", "content": user_input})

    response_generator = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        stream=True
    )
    response_list = list(response_generator)
    full_response = ""

    for item in response_list:
        content = item['choices'][0]['delta'].get('content', '')
        full_response += content
    return full_response

def display_chat_history(chat_history, character):
    for message, sender in chat_history:
        if ((sender == "Human")|(sender == "user")):
            color = "#3C4043"  # Dark gray for user
            bubble_css = "display: inline-block; background-color: {}; border-radius: 10px; padding: 6px 20px; margin-bottom: 2px; color: white; text-align: right".format(color)
            st.markdown(
                f'''
                <div style="text-align: right;">
                    <div style="color: #B0B0B0; font-size: 0.8em; font-weight: bold">{sender}</div>
                    <div style='{bubble_css}'>{message}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:  # Bot
            color = "#660000"  # Dark blue for bot
            bubble_css = "display: inline-block; background-color: {}; border-radius: 10px; padding: 6px 20px; margin-bottom: 2px; color: white".format(color)
            st.markdown(
                f'''
                <div style="text-align: left;">
                    <div style="color: #6C757D; font-size: 0.8em; font-weight: bold">{sender}</div>
                    <div style='{bubble_css}'>{message}</div>
                </div>
                ''', unsafe_allow_html=True)