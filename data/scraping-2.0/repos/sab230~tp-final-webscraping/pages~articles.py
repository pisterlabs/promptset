import streamlit as st
import openai

class TextProcessor:
    def __init__(self, api_key="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"):
        openai.api_key = api_key

    def openai_translate(self, text, target_language="fr"):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Translate the following English text to {target_language}: '{text}'",
            max_tokens=50
        )
        return response['choices'][0]['text'].strip()

    def openai_text_summary(self, text):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following text: '{text}'",
            max_tokens=100
        )
        return response['choices'][0]['text'].strip()

    def openai_text_generator(self, theme, content):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate text on the theme of '{theme}' with the following content: '{content}'",
            max_tokens=200
        )
        return response['choices'][0]['text'].strip()

    def openai_codex(self, code):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Correct the following code: '{code}'",
            max_tokens=150
        )
        return response['choices'][0]['text'].strip()

    def openai_image(self, prompt):
        response = openai.Completion.create(
            engine="image-alpha-003",
            prompt=f"Generate an image based on the following prompt: '{prompt}'",
            max_tokens=150
        )
        return response['choices'][0]['text'].strip()

    def openai_gpt3(self, user_input, context_url="https://www.blogdumoderateur.com/"):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"I am a chatbot trained to assist you. Ask me anything about articles from Blog du Modérateur: {context_url}\n\nUser: {user_input}\n",
            max_tokens=150
        )
        return response['choices'][0]['text'].strip()

api_key = "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"
text_processor = TextProcessor(api_key)

st.title("ChatGPT - Ask About Blog du Modérateur Articles")

# URL for context
blog_url = "https://www.blogdumoderateur.com/"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.markdown(f"{message['role']}: {message['content']}")

# React to user input
user_input = st.text_input("Ask me anything about articles from Blog du Modérateur")
if user_input:
    # Display user message in chat message container
    st.write(f"User: {user_input}")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process user input with OpenAI GPT-3
    assistant_response = text_processor.openai_gpt3(user_input, blog_url)

    # Display assistant response in chat message container
    st.markdown(f"ChatGPT: {assistant_response}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
