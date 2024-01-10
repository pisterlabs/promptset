# Empty VRAM cache
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
import time

#Import required libraries
import os 
from Openai_api import apikey 

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
#from langchain.agents import create_pandas_dataframe_agent
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

#from langchain.agents.agent_toolkits import create_python_agent

from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent

from dotenv import load_dotenv, find_dotenv

import torch
#from langchain import HuggingFacePipeline
#from langchain.llms.huggingface_pipeline import HuggingFacePipeline
#from transformers import AutoTokenizer, GenerationConfig, AutoConfig,pipeline

# from ctransformers import AutoModelForCausalLM
# from transformers import AutoTokenizer, GenerationConfig, AutoConfig,pipeline
# Load LLM and Tokenizer
# Use `gpu_layers` to specify how many layers will be offloaded to the GPU.
# model = AutoModelForCausalLM.from_pretrained(
#     "TheBloke/zephyr-7B-beta-GGUF",
#     model_file="zephyr-7b-beta.Q4_K_M.gguf",
#     model_type="mistral", gpu_layers=50, hf=True
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "HuggingFaceH4/zephyr-7b-beta", use_fast=True
# )

# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# text_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     use_cache=True,
#     tokenizer=tokenizer,
#     max_new_tokens=128
# )


os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

st.title("DataSage Insight 🤖")
st.write("Hello, 👋 I am DataSage and I am here to help you with data science projects.")

with st.sidebar:
    st.write('*Please upload your CSV files to Analyze*')
    st.caption('''**Discover hidden gems in your data with our powerful analytics and visualization tools.
    No PhD in data science required! Our intuitive interface ensures that everyone can navigate and analyze data like a pro.**
    ''')

    st.divider()
    # with st.expander('Expander section'):
    #     st.write('Test')

    st.caption("<p style ='text-align:center'> Open for Everyone..🎁</p>",unsafe_allow_html=True )


if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started!!", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    st.header('Data Analysis')
    st.subheader('Checking..')
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(temperature = 0)
        #llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
        #Function sidebar
        @st.cache_data
        def steps():
            steps_eda = llm('What are the steps of Data Analysis. Tell in short')
            return steps_eda

        #Testing
        pandas_agent=create_pandas_dataframe_agent(llm,df,verbose=True)
        # q='What is this data about?'
        # ans=pandas_agent.run(q)
        # st.write(ans)

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            # columns_df = pandas_agent.run("What are the meaning of the columns?")
            # st.write(columns_df)
            # missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            # st.write(missing_values)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            analysis = pandas_agent.run("What is maximum profit that I could have got? Explain how")
            st.write(analysis)
            # conc = pandas_agent.run("So what can you conclude from this data?.")
            # st.write(conc)
            # new_features = pandas_agent.run("What new features would be interesting to create? Just give some ideas.")
            # st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.bar_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            # trends = pandas_agent.run(f"Analyse trends, seasonality, or cyclic patterns of {user_question_variable}")
            # st.write(trends)
            # missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            # st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_resource
        def wiki(prompt):
            wiki_research = WikipediaAPIWrapper().run(prompt)
            return wiki_research

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )
            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
            )
            return data_problem_template, model_selection_template

        @st.cache_resource
        def chains():
            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
            return sequential_chain

        @st.cache_resource
        def chains_output(prompt, wiki_research):
            my_chain = chains()
            my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
            my_data_problem = my_chain_output["data_problem"]
            my_model_selection = my_chain_output["model_selection"]
            return my_data_problem, my_model_selection

        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split('\n')
            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output
        
        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )
            return agent_executor
        
        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):
            solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}, and handle all the exceptions raised efficiently."
            )
            return solution

        #Main

        st.header('Data analysis')
        st.subheader('General information about the dataset')

        # with st.sidebar:
        #     with st.expander('Steps of Data Analysis'):
        #         st.write(steps())

        function_agent()

        st.subheader('Parameter study')
        user_question_variable = st.text_input('What parameter are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")
                if user_question_dataframe:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Reframing the business problem into a data science problem...")
                    
                    prompt = st.text_area('What is the business problem you would like to solve?')
                              
                    if prompt:
                        wiki_research = wiki(prompt)
                        my_data_problem = chains_output(prompt, wiki_research)[0]
                        my_model_selection = chains_output(prompt, wiki_research)[1]
                            
                        st.write(my_data_problem)
                        st.write(my_model_selection)

                        formatted_list = list_to_selectbox(my_model_selection)
                        selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                        if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                            st.subheader("Solution")
                            solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                            st.write(solution)