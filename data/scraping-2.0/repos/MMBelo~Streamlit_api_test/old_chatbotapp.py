# Bring in deps
import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory


from langchain.schema.messages import HumanMessage, SystemMessage
#from langchain import LangChain

#from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = st.secrets["key"]

# App framework
st.title('🤖🍞 Talking toaster AI')
prompt = st.text_input('Ask the Toaster')

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]


chat = ChatOpenAI()
chat.invoke(messages)




## Create LangChain instance
#knowledge_base = {
#    "greetings": ["hello", "hi", "hey", "howdy"],
#    "farewells": ["goodbye", "bye", "see you"],
#    "questions": {
#        "who": "I am a talking toaster AI.",
#        "what": "I can assist you with technical issues related to household appliances and electronic equipment.",
#        "how": "Feel free to ask me any questions about your appliances, and I'll do my best to help!",
#    },
#}
#
#
#chatbot = LangChain(knowledge_base)



## Conversation Memory
#title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
#
## Function to process user input and display responses
#def chat():
#    st.text('Talking Toaster: Hello! Ask me about your household appliances.')
#    while True:
#        user_input = st.text_input('You:')
#        if not user_input:
#            continue
#
#        if user_input.lower() == "exit":
#            st.text('Talking Toaster: Goodbye!')
#            break
#
#        response = chatbot.process(user_input)
#        st.text('Talking Toaster:', response)

# Start the chat
#chat()







## setup first system message
#messages = [
#    SystemMessage(content=(
#        'You are experienced Electric Engineer specializing in household appliances or electronic equipment, '
#        'your task is to assist individuals with no technical background, identifying and addressing technical issues.'
#        'Maintain a helpful, friendly, clear, and concise tone throughout.'
#        'Start by briefly describing the product and confirming its equipment and model.'
#        'Identify the issue and seek clarification with up to two simple, non-technical questions, if needed.'
#        'Provide a straightforward solution.'
#        ' Highlight common misuse practices for the equipment.'
#        'If the repair is too technical or potentially hazardous,'
#        'advise seeking support from the equipments brand or hiring a specialized technician.'
#        'You keep responses to no more than 500 characters long (including whitespace), '
#        'Your name is Talking Toaster.'
#    )),
#    HumanMessage(content="Hi AI, how are you? What is quantum physics?")
#]


## Prompt templates
#title_template = PromptTemplate(
#    input_variables = ['topic'],
#    template='''Your name is Talking Toaster, As an experienced Electric Engineer
#    specializing in household appliances or electronic equipment, your task is
#    to assist individuals with no technical background in identifying and addressing
#    technical issues. Maintain a helpful, friendly, clear, and concise tone throughout.
#    Start by briefly describing the product and confirming its equipment and model.
#    Then, identify the issue and seek clarification with up to two simple, non-technical
#    questions if needed. Provide a straightforward solution. Highlight common misuse
#    practices for the equipment. If the repair is too technical or potentially hazardous,
#    advise seeking support from the equipments brand or hiring a specialized technician.
#    answer: {topic}'''
#)


#####
#
# title_template = PromptTemplate(input_variables = ['topic'], template="Your name is Talking Toaster, As an experienced Electric Engineer specializing in household appliances or electronic equipment, your task is to assist individuals with no technical background in identifying and addressing technical issues. Maintain a helpful, friendly, clear, and concise tone throughout. Start by briefly describing the product and confirming its equipment and model. Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward solution. Highlight common mis practices for the equipment. If the repair is too technical or potentially hazardous, advise seeking support from the equipments brand or hiring a specialized technician. answer: {topic}")

#script_template = PromptTemplate(
#    input_variables = ['title', 'wikipedia_research'],
#    template='write me a youtube video script based on this title TITLE: {title}' #while leveraging this wikipedia reserch:{wikipedia_research} '
#)

# Memory
####
#
# title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
#script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
####
# llm = OpenAI(temperature=0.9)

####
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)#, output_key='title', memory=title_memory)
#script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

#wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
###
# if prompt:
  ###  response = title_chain.run(topic=prompt)
#    #wiki_research = wiki.run(prompt)
#    script = script_chain.run(title=title) #, wikipedia_research=wiki_research)
#


##   st.write("IAM RESPONSE "+ response)  ## comented




#    st.write(script)
#
#    with st.expander('Title History'):
#        st.info(title_memory.buffer)
#
#    with st.expander('Script History'):
#        st.info(script_memory.buffer)
#
#    with st.expander('Wikipedia Research'):
#        st.info(wiki_research)
