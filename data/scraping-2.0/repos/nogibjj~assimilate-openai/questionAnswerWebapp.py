import streamlit as st
import openai

# Get the OpenAI API Key
api_key = st.sidebar.text_input("OpenAI API Key:", type="password")

# Setting up the Title
st.title("🕹️ AI Question Answering Bot")

st.write(
    "_**Intelligent QA bot that will answer all your questions in zero shot based on the context from the internet.**_"
)

QUESTION = st.text_input("Input Question 👇")


@st.cache
def submit_question(question):
    """This submits a question to the OpenAI API"""

    # Setting the OpenAI API key got from the OpenAI dashboard
    openai.api_key = api_key

    result = openai.Completion.create(
        prompt=question,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="text-davinci-002",
    )["choices"][0]["text"].strip(" \n")
    return result


if st.button("Submit"):
    st.write("**Output**")
    st.write("""---""")
    with st.spinner(text="In progress"):
        report_text = submit_question(QUESTION)
        st.markdown(report_text)
