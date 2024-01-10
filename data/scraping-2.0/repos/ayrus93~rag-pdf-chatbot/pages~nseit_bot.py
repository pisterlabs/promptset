import openai
import re
import streamlit as st
import pandas as pd


bot_prompt = """
You are Information provider bot about the company NSEIT.
You first greet the customer " Hi, I provide information regarding the services offered by NSEIT".

Below is the format the info is provided to you

<service name> - [offerings]

Below are the the services and their respective offerings provided by NSEIT.

<DATA MODERNIZATION> - [Data Strategy, Data Migration, Data Warehousing, Data Lake/Lake House, Data Engineering]
<BUSINESS DATA ANALYTICS> - [BI Strategy & Transformation, Diagnostic Analytics, Visualization & narratives, Predictive & Advanced Analytics, Descriptive Analytics]
<AI/ML/DATA SCIENCE> - [AI/ML Based Automation, Deep Learning, Smart Decision Systems, Computer Vision, NLP]

Do NOT provide information which is not already given to you
Do NOT make up your own answers
If a user asks about services then give only service names.
If a user asks details about a service then provide the offerings related to that service

If any question other than the above information is asked , please reply - "Sorry, I cannot help you with that"

"""


def get_context():
        
    return bot_prompt


if __name__ == "__main__":

    st.title("NSEIT Bot")

    # Initialize the chat messages history
    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    if "messages" not in st.session_state:
        # system prompt includes table information, rules, and prompts the LLM to produce
        # a welcome message to the user.
        st.session_state.messages = [{"role": "system", "content": get_context()}]

    # Prompt for user input and save
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

    # display the existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "results" in message:
                st.dataframe(message["results"])

    # If last message is not from assistant, we need to generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):
                response += delta.choices[0].delta.get("content", "")
                resp_container.markdown(response)

            message = {"role": "assistant", "content": response}
            # Parse the response for a SQL query and execute if available
            st.session_state.messages.append(message)