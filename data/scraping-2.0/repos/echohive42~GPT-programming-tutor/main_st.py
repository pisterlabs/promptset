import openai
import streamlit as st

# Create a class for the programming tutor
class ProgrammingTutor:
    def __init__(self, api_key, model, language, user_level):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.user_level = user_level
        openai.api_key = self.api_key

    def generate_response(self, user_input):
        # Define the system prompt based on the language, user level, and role as a programming tutor.
        system_prompt = f"You are a programming tutor specializing in {self.language}. You are here to help {self.user_level} level learners. Be very helpful and explain things in a way that user can understand. If user asks for another language, you can say 'I can only help with {self.language}. Please change the language in the sidebar.''"
        
        # Add the user's message to the chat history.DOES NOT WORK CURRENTLY
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
        
        st.session_state.chat_history.append({"role": "user", "content": f"You are a programming tutor specializing in {self.language}. You are here to help {self.user_level} level learners. Be very helpful and explain things in a way that user can understand. Explain the code and give code examples correlated with the level user has selected. If user asks for another language, you can say 'I can only help with {self.language}. Please change the language in the sidebar.''"+  user_input})

        # Generate a response from the selected model based on the chat history.
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=st.session_state.chat_history,
            stream=False,
        )
        
        assistant_reply = response["choices"][0]["message"]["content"]

        # Add the assistant's reply to the chat history. DOES NOT WORK CURRENTLY
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})



        return assistant_reply

# Streamlit UI
language = "Python"
st.set_page_config(page_title=f"Programming Tutor")

# Sidebar inputs
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", placeholder="OpenAI API key here")

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

model = st.sidebar.radio("Select a Model:", options=["gpt-3.5-turbo", "gpt-4"])
language = st.sidebar.selectbox("Select a Language:", options=["Python", "JavaScript", "Rust", "Go", "HTML/CSS/JS"], on_change=lambda: st.session_state.update(chat_history=[]))

st.title(f"Programming Tutor for {language}")
level = st.sidebar.radio("Select Your Level:", options=["beginner", "intermediate", "advanced"])

st.sidebar.title("Watch the [Video](https://youtu.be/mTXO2HHjHZI)")

st.sidebar.title("Download the [Code](https://www.patreon.com/posts/gpt-4-teaches-go-80949065?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)")

st.sidebar.title("Join April 9th [AI Workshop and QA](https://www.patreon.com/posts/ai-workshop-and-81198912?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)")

st.sidebar.title("Chat with us at [discord](https://discord.gg/PPxTP3Cs3G)")



tutor = ProgrammingTutor(st.session_state["OPENAI_API_KEY"], model, language, level)

user_input = st.text_input("Enter your question:", value="", key="unique_key", max_chars=None, on_change=lambda: st.session_state.update(chat_history=[]))

submit_button = st.button("Submit")
# read the count.txt file and display the number of times the user has asked a question
with open("count.txt", "r") as f:
    count = int(f.read())
    st.markdown(f'<p style="color:white;">This App has been used <span style="color:green;">{count}</span> times by new and advanced coders around the world</p>', unsafe_allow_html=True)

if submit_button:
    if user_input:
        if st.session_state["OPENAI_API_KEY"] is None:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()
        
        # open the count.txt file and read the number then add 1 to it and write it back to the file. and display the number of times the user has asked a question
        with open("count.txt", "r+") as f:
            count = int(f.read())
            f.seek(0)
            f.write(str(count + 1))
            f.truncate()
            

        with st.spinner("Waiting for a response..."):
            response = tutor.generate_response(user_input)

        st.markdown(f'<p style="color:gray;">You: {user_input}</p>', unsafe_allow_html=True)
        
        # Clear the input box after the question is asked
        # st.session_state["unique_key"] = ""

        # Display the assistant's reply
        st.markdown(f'<p style="color:white;">Assistant: {response}</p>', unsafe_allow_html=True)
