import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' >EcoKids Hub Doubt Solver 🤔</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Unlocking the Secrets of Nature: Your EcoKids Hub Doubt Solver!</h3>", unsafe_allow_html=True)

prompt = st.text_input('Have a doubt 🤯! Ask here🧠') 

doubt_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Solve doubt for kids on {topic}'
)

# Llms
llm = OpenAI(temperature=0.9) 


doubt_chain = LLMChain(llm=llm, prompt=doubt_template, verbose=True, output_key='doubt')


# Show stuff to the screen if there's a prompt
if prompt: 
    doubt = doubt_chain.run(prompt)
    
    st.write(doubt) 