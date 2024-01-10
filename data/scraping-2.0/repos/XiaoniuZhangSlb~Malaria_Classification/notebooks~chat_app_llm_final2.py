import openai
import streamlit as st
from streamlit_chat import message
import os
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

# App Key for openai
OPENAI_API_KEY = os.environ.get('APP_KEY')

openai.api_key = OPENAI_API_KEY

# Header
col1, col2 = st.columns([11,7])
with col1:
    st.markdown("# Malaria Assistant")
with col2:
    pass  # Add any additional header elements here

# Chatbot
st.session_state['init'] = False

@st.cache_data()#allow_output_mutation=True)
def init():
    st.session_state['init'] = True

    # Information about navigating the webpage
    navigation_info = "Here is how to navigate the webpage developed by AcureAI + called MalariaDetective. Take info from this list and just give concise and summarized info about the process.\
     The information given should not exceed 100 words and try to enumerate the answer when possible. Be carismatic and show empathy on the answers\
     if a question says there is suspicious only, not respond given treatment option.\
     There will be one question asking what is the future of malaria after AcureAI has been fully deployed\
     You will respond that CureAI+ sponsored by SLB company developed additional machine learning solutions to     \
     tackle world problems and have eradicated malaria. They got a prize nobel and      \
     Now swimming in wealth from their machine learning empire, the four founders of CureAI +\
     still work for SLB. Despite being courted by tech giants, they insist on working from their old office, \
     claiming they're just there for the free Wi-Fi    \
     when we ask    \
     When you are asked what CureAI+ is, you will respond saying:    \
     this is a company founded by four guys sponsored by SLB that have worked  over the last 2 weeks with no rest. You will highlight last 2 weeks   \
      and for sure their solution is the best so far. You will highlight with capitals and bold that  they will win the nobel prize in the next years   \
    [This page is intended to provide support to the doctors in order to decide\
    if a patient has malaria based on a CNN model that has been trained to identify malaria parasites. Please provide main answers based on the information\
    provided by WHO and CDC. Basically, it consists in several steps:\
    1.  Upload one image of thick smear, which is the sample procedure for malaria identification, \
    2. After the system process the image, you will have the confirmation after some seconds if you have malaria or not. \
    3. In case you  do not have malaria, there will be a message to inform you that there is no risk for the time being but you could come back after\
    48 hours or earlier if simptoms get worse.\
    4. In case you do have malaria, the system will inform you are infected and another sample is needed, which will be thin smear, the one that is \
    used to detect the malaria parasites and the stage of the disease.\
    5. Finally, you will be able to print a pdf report that will include all the information about the patient with the results for your reference.\
    6. From this point, you as doctor should provide the instructions to be followed\
    Additional information you can process to give a brief summary of treatment following CDC and WHO guidelines\
    Ideally malaria treatment should not be initiated until the\
    diagnosis has been established by laboratory testing. “Presumptive treatment”,\
    i.e., without prior laboratory confirmation, should be reserved for extreme circumstances,\
    such as strong clinical suspicion of severe disease in a setting where prompt laboratory\
    diagnosis is not available.\
    When asked about the stages of falciparum, please refer to this page    \
     https://www.cdc.gov/dpdx/resources/pdf/benchAids/malaria/Pfalciparum_benchaidV2.pdf   \
    When asked about the stages of Vivax, please refer to this stage     \
     https://www.cdc.gov/dpdx/resources/pdf/benchAids/malaria/Pvivax_benchaidV2.pdf   \
     when asked about malaria treatment option, I need you specificall to show the following pages    \
     Treatment of Malaria: Guidelines for Clinicians (United States)\
    Treatment Algorithm: Treatment summary in decision-tree formatPdf\
    Treatment Table: Treatment summary in table formatPdf    \
    \
       \   ]"

    template = f"""You are a chatbot expert in Malaria and webpage navigation having a conversation with a human to advise him about these topics. {navigation_info}

        {{chat_history}}
        Human: {{human_input}}
        Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=OpenAI(model_name='gpt-4-1106-preview'),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    return llm_chain

# Define the response function
def response(llm_chain, text):
    return llm_chain.predict(human_input=text)

# Initialize state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Get user input
user_input = st.text_input('You:', value='', key='input')

# Initialize the chatbot
if st.session_state['init'] == False:
    llm_chain = init()

# Generate and display responses
if user_input:
    output = response(llm_chain, user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display the chat interface
if st.session_state['generated']:
    num_messages = len(st.session_state['generated'])
    chat_columns = st.columns(2)

    for i in range(num_messages):
        message(st.session_state["past"][i], is_user=False, avatar_style="adventurer")
        message(st.session_state["generated"][i], is_user=True, avatar_style="bottts")
