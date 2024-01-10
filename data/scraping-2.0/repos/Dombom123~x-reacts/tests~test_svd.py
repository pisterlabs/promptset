import streamlit as st
import replicate as rep
from openai import OpenAI

def get_assistant_response(prompt):
    api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )

    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
    )
    
    thread_messages = client.beta.threads.messages.list(thread.id)
    print(thread_messages.data)
    
    return thread_messages.data
    

def generate_video(prompt):
    get_assistant_response(prompt)




def main():
    st.set_page_config(page_title="X-Reacts", page_icon="🎥")
    st.title("X-Reacts Video Generation 🎥")
    st.image("data/header.png")

    prompt = st.text_area("Prompt", "This is a test prompt.")
    if st.button("Generate"):
        result = get_assistant_response(prompt)
        st.write(result)
        st.success("Video generated!")



if __name__ == "__main__":
    main()