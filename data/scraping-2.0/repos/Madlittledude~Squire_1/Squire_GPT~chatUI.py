import streamlit as st
import openai
import os
# Set the page config here


def chatbot_ui_page():
   

    # Function to display chat messages
    @st.cache_data()
    def display_chat_message(role, content, avatar=None):
        with st.chat_message(role, avatar=avatar if avatar else None):
            st.markdown(content)

    def display_intro():
        st.title('Brain_Storm :lightning_cloud:')
        st.write("This tool is not a factbook. In essence, this tool is a reflection of your input bouncing off a large language model at the angle of your perspective.")
        st.write("Squire's function is to facilitate the brainstorming process. Think of it as if you're the Knight and it's your literate Squire.")
        st.write("We're using a large language model (Specifically: OpenAi's GPT chat models) which allow us to engage in a digitally augmented train of thought. It's like having a conversation with yourself, but with the added advantage of tapping into vast textual patterns and diverse perspectives.")
        st.write("Its effectiveness and output are directly influenced by the clarity and specificity of your questions. You'll understand the more you use it. Just remember, it's a reasoning/ structuring tool, not a factbook.")
        st.write("With your domain knowledge use this prompt to get an idea of what you can expect from Squire: How do you draft a {insert legal doc}?")
        st.write('  :heart: Max')
    
    def display_chat_interface():
       
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            avatar = "Squire_GPT/ASSETS/Madlittledude 2_white.png" if message["role"] == "assistant" else "Squire_GPT/ASSETS/madlittledude_flipped.png"
            display_chat_message(message["role"], message["content"], avatar)

        # User input
        prompt = st.chat_input("Start thinking with your fingers...get your thoughts out")
        if prompt:
            # Set the state to indicate the user has sent their first message
            st.session_state.first_message_sent = True
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt, avatar="Squire_GPT/ASSETS/madlittledude_flipped.png")

            with st.chat_message("assistant", avatar="Squire_GPT/ASSETS/Madlittledude 2_white.png"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=([
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]),
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
  

    # Initialization logic
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": ("You're Squire_GPT and you work at a municipal law firm, ...")
        }]

    if "first_message_sent" not in st.session_state:
        st.session_state.first_message_sent = False

    openai.api_key = os.environ['OPENAI_API_KEY']


    # Display logic
    if not st.session_state.first_message_sent:
        display_intro()

    display_chat_interface()

if __name__ == "__main__":
    chatbot_ui_page()
