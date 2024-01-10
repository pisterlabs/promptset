import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.agents import AgentType

st.title("LeanStartupAgent")

st.write("This agent executes the basic steps of the Lean Startup Approach for developing novel business models. As a user, you need to enter your job-to-be-done and you need to give a description of your target customer segment. Subsequently, the agent will generate (i) Description of the core painpoint, (ii) Description of potential value proposition, (iii) Overview of value proposition canvas and (iv) Core hypotheses to be tested.")

st.write("The theoretical grounding of this agent is based on the work of people such as Steve Blank (https://mostawesomepodcast.com/tag/steve-blank/), Alex Osterwalder (https://mostawesomepodcast.com/strategyzer/) and Eric Ries.")

#define company and project

open_api_key = st.text_input('Enter your open api key. This information is not recorded or stored in any way', type = "password")

job_to_be_done = st.text_input('Describe the core job to be done')
customer_description = st.text_input('Describe your customer segment')

clicked = st.button('Click me')

if clicked:
    st.write('Button clicked! Performing an operation...')
    # Place the code that should only execute after click here
    llm = OpenAI(openai_api_key=open_api_key, temperature=0.5)
    hardest_template = PromptTemplate(
      input_variables = ['customer_description','job_to_be_done'], 
      template='You are {customer_description}. What is the hardest part about {job_to_be_done}?')
    hardest_chain = LLMChain(llm=llm, prompt=hardest_template, verbose=True, output_key='hardest_part')
    hardest = hardest_chain.run(customer_description=customer_description, job_to_be_done=job_to_be_done)
    st.markdown('**Agent core painpoint =**')
    st.write(hardest)
    value_proposition_template = PromptTemplate(
      input_variables = ['hardest_part'], 
      template='Create a unique value proposition for a startup that solves the problem of {hardest_part}')
    value_proposition_chain = LLMChain(llm=llm, prompt=value_proposition_template, verbose=True, output_key='value_proposition')
    valueproposition = value_proposition_chain.run(hardest_part=hardest)
    st.markdown('**Agent Value proposition =**')
    st.write(valueproposition)
    canvas_template = PromptTemplate(
      input_variables = ['value_proposition'], 
      template='Provide the value proposition canvas for {value_proposition}. You should describe the following elements: customer jobs, custome pains, customer gains, products and services, pain relievers, gain creators')
    canvas_chain = LLMChain(llm=llm, prompt=canvas_template, verbose=True, output_key='canvas')
    canvas = canvas_chain.run(value_proposition=valueproposition)
    st.markdown('**Agent description of value proposition canvas =**')
    st.write(canvas)
    # prompting hypotheses
    hypotheses_template = PromptTemplate(
      input_variables = ['value_proposition', 'canvas'], 
      template='Generate a list of hypotheses for testing the following value proposition: {value_proposition}. You should rely on information from the value proposition canvas: {canvas}. Include at least 3 hypotheses. Each hypothesis should focus on the feasibility or viability of the business model. Here are some potential topics for hypotheses testing: willingness to pay, preferred distribution model, ultimate customer segmet, etc.')
    hypotheses_chain = LLMChain(llm=llm, prompt=hypotheses_template, verbose=True, output_key='hypotheses')
    hypotheses = hypotheses_chain.run(value_proposition=valueproposition, canvas=canvas)
    st.markdown('**Agent hypotheses =**')
    st.write(hypotheses)
else:
    st.write('Please click the button to perform an operation')
