import openai
import os
import streamlit as st
from APIs.text2speech import get_speech_from_text
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from pathlib import Path
openai.api_key  = os.getenv('OPENAI_API_KEY')
from APIs.helper import autoplay_audio

def ask_openai(messages, temperature=0.5):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return "I'm sorry, but I can't answer that question."

def chat_with_child(user_message):
    print("Hello! Welcome to GabbyGarden! I am Gab.\n You can ask me any question you like! I'm here to help you learn and understand new things.")

    conversation_log = [] # to maintain the conversation context
    # while True:
        # user_message = input("What's your question? (type 'exit' to stop) ")
    # if user_message.lower() == 'exit':
    #     break

    # Add context for simplicity and a child-friendly tone
    # context = "Please respond in a way that a five-year-old would understand. "
    context = "You are a chatbot that talks to young kids.\n\
            Some requirements are as follows: \n\
            1. You use easy words and short sentences to make sure they can understand you. \n\
            2. Your job is to answer questions, tell stories, and share fun facts about things like animals, space, and cartoons in a safe and friendly way.\n\
            3. You always keep the conversation appropriate for kids and never include anything that isn't safe for them to hear. \n\
            4. If the kid asks you a inproper question, you should refuse the answer and elaborate why."


    # print('====================context====================')
    # print(context)
    # print('===============================================')
    # Add the user's message with context to the conversation log
    conversation_log.append({"role": "system", "content": context})
    conversation_log.append({"role": "user", "content": user_message})

    # Check for the word "bad" in the question as a naive demonstration of content filtering
    if "bad" in user_message.lower(): # Replace with a more robust filter or moderation approach
        bot_response = "Let's try to keep our conversation positive and educational! Do you have another question?"
        # continue
    else:
        # Get the chatbot's response and print it
        bot_response = ask_openai(conversation_log)
        # print("Answer:", bot_response)

    # Add the chatbot's response to the conversation log
    conversation_log.append({"role": "assistant", "content": bot_response})
    get_speech_from_text(bot_response)
    # audio_path = str(Path(os.getcwd()) / Path("assets/text2speech.mp3"))
    # st.audio(audio_path)
    # autoplay_audio("./assets/text2speech.mp3")
    
    return conversation_log
        # flag = input("Do you want me to tell you more about this topic? (y/n)")

        # if flag.lower() == 'n':
        #     continue
        # else:
        #     conversation_log.append({"role": "user", "content": "Can you please tell me more about this topic?"})
        #     bot_response = ask_openai(conversation_log)
        #     print("Answer:", bot_response)
        #     conversation_log.append({"role": "assistant", "content": bot_response})
        #     continue



if __name__ == "__main__":
    chat_with_child()
