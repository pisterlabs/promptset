import streamlit as st
from openai import OpenAI
from colorama import Fore, Style
import time
from check_run import check_run

client = OpenAI(api_key=st.secrets.openai.api_key_general)

def chat_loop(vAssistantId, vThreadId):

    while True:

        #input from user
        user_input = input(f"{Fore.CYAN} User: ")
        print(Style.RESET_ALL)
        if user_input == "quit":
            break

        message = client.beta.threads.messages.create(
            thread_id=vThreadId,
            role="user",
            content=user_input
        )

        run = client.beta.threads.runs.create(
            thread_id=vThreadId,
            assistant_id=vAssistantId
        )

        check_run(vThreadId, run.id)

        #get latest messages from the thread
        messages = client.beta.threads.messages.list(
            thread_id=vThreadId
        )

        #get the latest from user
        user_message = messages.data[1].content[0].text.value

        #get latest from assistant
        assistant_message = messages.data[0].content[0].text.value

        # Print the latest message from the user
        # print(f"{Fore.CYAN} User: {user_message} {Style.RESET_ALL}")

        # Print the latest message from the assistant
        print(f"{Fore.BLUE} Assistant: {assistant_message} {Style.RESET_ALL}")

#https://github.com/davideuler/awesome-assistant-api/blob/main/GPT-Assistant-Tutoring.ipynb