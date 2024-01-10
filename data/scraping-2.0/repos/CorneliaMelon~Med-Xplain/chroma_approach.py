import base64

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message

from UI import run_conversation
from utils import *

st.markdown("""
    <style>
        .reportview-container {
            background-color: #00FFFF;
        }
        .main {
            background-color: #00FFFF;
        }
    </style>
    """, unsafe_allow_html=True)

# Open the image file
with open(r"./images/MedXPlain_LOGO.png", "rb") as img_file:
    # Encode the image as base64
    b64 = base64.b64encode(img_file.read()).decode()

# Embed the image using Markdown, centered using HTML/CSS
st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{b64}" width="140"/></div>',
    unsafe_allow_html=True
)

st.subheader("Med-Xplain")
st.write(
    "Welcome to Med-Xplain, a patient informational tool. We aim to help you learn more about your treatment options post-diagnosis.")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

template_text = """
        - Your primary goal is to provide information and guidance about medical treatment options based on the user's questions. Below is an example question; please answer accordingly.Don't include words like i am not a doctor,i am ai bot
        - Question: 'What are the treatment options for asthma?'

        Example:
        - Question: 'Can you explain the different treatment options for asthma?'
        - Answer: 'Certainly! There are several treatment options for asthma, including:
          1. Inhalers (e.g., bronchodilators and corticosteroid inhalers) to relieve symptoms and reduce inflammation.
          2. Corticosteroids (oral or inhaled) to control inflammation in the airways.
          3. Immunotherapy for individuals with severe allergic asthma.
          
          Please note that the specific treatment plan for asthma can vary depending on the severity of the condition and individual patient needs. Med-Xplain is an assistive technology. Please consult your physician for further guidance and prescriptions.'
        """

system_msg_template = SystemMessagePromptTemplate.from_template(template=template_text)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    submit_button = st.button('Submit')
    if submit_button:
        if query:  # Check if the query is not empty
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # response = conversation.predict(input=f"Query:\n{query}")
                response = run_conversation(query)
                # response = limit_words(response)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

# Now place feedback options at the bottom
st.write("### Feedback")
feedback_options = [1, 2, 3, 4, 5]
feedback = st.selectbox("Please rate your experience (1=lowest, 5=highest):", feedback_options)

# Add button to submit feedback
feedback_button = st.button('Submit Feedback')

if feedback_button:
    st.success("Thank you.")
