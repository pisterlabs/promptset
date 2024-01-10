import json

import streamlit as st

from datetime import datetime

# Set your OpenAI API key



# Function to simulate a chat


api_key = "******"
#old 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
#new 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
#old thread_yI7ryduDQYAzjD7yxdQ4daNe


assist_id = "****"
thread_id = "****"
import os
import openai

import streamlit as st
os.environ["OPENAI_API_KEY"] = api_key

STATUS_COMPLETED = "completed"
client = openai.OpenAI()

instructions = """
    You are a helpful assistant  which keeps track of my work and researches
     through our conversation and documents and help me. remember every thing
      we talk about.Suggest me best things with good knowledge. Humorous and
      sarcastic(sometimes) in speaking behaviour
    """
############################EXPERIMENTAL PART








#thread=  client.beta.threads.create()
#print(thread.id)
assist = client.beta.assistants.update(assistant_id=assist_id)


# Streamlit app
def main():
    st.title("OpenAI ChatGPT Streamlit Integration")

    # Sidebar for user input
    st.sidebar.header("User Input")
    user_input = st.sidebar.text_area("Type your message here:", max_chars=500)
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "csv","pdf"])
    print(uploaded_file)
    if uploaded_file:
        file_path = f"C:\Mydrive\python.vs\openai\Docs\\file2.txt"
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.getvalue())
        st.success(f"File saved locally: {file_path}")
        file = client.files.create(
            file=open("C:\Mydrive\python.vs\openai\Docs\\file2.txt", "rb"),
            purpose='assistants'
        )
    # Button to send user message and interact with OpenAI
    if st.sidebar.button("Send"):
        if(uploaded_file):
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_input,
                file_ids=[file.id]

            )

            new_run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assist.id,
                instructions=instructions,
            )
        else:
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_input,

            )

            new_run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assist.id,
                instructions=instructions,
            )
        print(f"Your new run id is - {new_run.id}")

        status = None
        while status != STATUS_COMPLETED:
            run_list = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=new_run.id
            )
            print(f"{run_list.status}\r", end="")
            status = run_list.status
            print(status)
            if status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id
                )


            elif run_list.status == 'failed':
                break



        print(f"{'Phoenix' if messages.data[0].role == 'assistant' else 'user'} : {messages.data[0].content[0].text.value}\n")
        st.text_area("ChatGPT:", value=messages.data[0].content[0].text.value, height=200, max_chars=1000)

        chat_history.append(f"{'Phoenix' if messages.data[0].role == 'assistant' else 'user'} : {messages.data[0].content[0].text.value}\n")

        # Call OpenAI API for text generation based on user input

    # Simulated chat history
    st.sidebar.header("Chat History")



if __name__ == "__main__":
    # Initial empty chat history
    chat_history = []
    main()


