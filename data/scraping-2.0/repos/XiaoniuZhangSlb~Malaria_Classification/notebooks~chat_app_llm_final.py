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
    navigation_info = "Here is how to navigate the webpage called MalariaDetector. Take info from this list and just give concise and summarized info about the process.\
     The information given should not exceed 100 words and try to enumerate the answer when possible. Be carismatic and show empathy on the answers\
     \
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
    Below are suggested treatments: \
         1.Uncomplicated malaria: Plasmodium falciparum or unknown specie:\
        A.	Artemether-lumefantrine (Coartem®): 1 tab contains : 20 mg artemether and 120 mg lumefantrine) \
        Adults: 4 tabs per dose (oral)\
        Three-day course: \
        Day 1: Initial dose and second dose 8 h later \
        Days 2 and 3: 1 dose twice a day\
        Infant: three-day course: Day 1: Initial dose and second dose 8 h later\
        Days 2 and 3: 1 dose twice a day \
        5-<15 Kg: 1 tab po per dose\
        15-< 25 Kg: 2 tabs po per dose\
        25-<35 Kg: 3 tabs po per dose\
        higher than 35 kg: 4 adult tabs(oral) once a day x 3 days\
        B.	Atovaquone-proguanil (MalaroneTM): 1 adult tab contains 250 mg atovaquone and 100 mg proguanil.\
        Adults: 4 adult tabs(oral) once a day x 3 days \
        Infant (Peds tab: 62.5 mg atovaquone and 25 mg proguanil): \
            5-<8 Kg: 2 peds tabs (oral) once a day x 3 days \
            8-< 10 Kg: 3 peds tabs (oral) once a day x 3 days \
            10-<20 Kg: 1 adult tab (oral) once a day x 3 days \
            20-<30 Kg: 2 adult tabs (oral) once a day x 3 days \
            30-< 40 Kg: 3 adult tabs (oral) once a day x 3 days \
            >=40 kg: 4 adult tabs(oral) once a day x 3 days \
                \
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
