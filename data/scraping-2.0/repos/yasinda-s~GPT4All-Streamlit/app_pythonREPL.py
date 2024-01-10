import streamlit as st

#import dependencies
from langchain.llms import GPT4All #you can also use OpenAI, HuggingFace Hub, or GPT4All
from langchain import PromptTemplate, LLMChain

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

#path to weights
PATH = 'D:/gpt4all/weights/ggml-model-gpt4all-falcon-q4_0.bin' #using the falcon q4 model, snoozy is probably the best from GPT4ALL
llm = GPT4All(model=PATH, verbose=True) #load model

# create the python agent executor
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)


st.title("Langchain GPT4All")

prompt_from_user = st.text_input("Add any prompt here!")

if prompt_from_user:
    #pass the user prompt to python agent executor
    response = agent_executor.run(prompt_from_user)
    #return the response
    st.write(response)

