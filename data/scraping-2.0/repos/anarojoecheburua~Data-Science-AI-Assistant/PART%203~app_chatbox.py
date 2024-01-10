#Import required libraries
import os 
from apikey import apikey 

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from streamlit_chat import message
from auxiliary_functions import *








#OpenAIKey
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

#Title
st.title('AI Assistant for Data Science 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

#Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made with ❤️ by Ana</p>",unsafe_allow_html=True )


#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    tab1, tab2 = st.tabs(["Data Analysis and Data Science","ChatBox"])
    with tab1:
        user_csv = st.file_uploader("Upload your file here", type="csv")
        if user_csv is not None:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory=False)

            #llm model
            llm = OpenAI(temperature = 0)

            #Function sidebar
            @st.cache_data
            def steps_eda():
                steps_eda = llm('What are the steps of EDA')
                return steps_eda
            
            @st.cache_data
            def data_science_framing():
                data_science_framing = llm("Write a couple of paragraphs about the importance of framing a data science problem approriately")
                return data_science_framing
            
            @st.cache_data
            def algorithm_selection():
                data_science_framing = llm("Write a couple of paragraphs about the importance of considering more than one algorithm when trying to solve a data science problem")
                return data_science_framing

            #Pandas agent
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

            #Functions main
            @st.cache_data
            def function_agent():
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.write(df.head())
                st.write("**Data Cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
                duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                st.write(duplicates)
                st.write("**Data Summarisation**")
                st.write(df.describe())
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
                new_features = pandas_agent.run("What new features would be interesting to create?.")
                st.write(new_features)
                return

            @st.cache_data
            def function_question_variable():
                st.line_chart(df, y =[user_question_variable])
                summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
                st.write(summary_statistics)
                normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                st.write(normality)
                outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                st.write(outliers)
                trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                st.write(trends)
                missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                st.write(missing_values)
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

            @st.cache_data
            def chains():
                data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
                model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
                sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
                return sequential_chain

            @st.cache_data
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
                solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
                )
                return solution

            #Main

            st.header('Exploratory data analysis')
            st.subheader('General information about the dataset')

            with st.sidebar:
                with st.expander('What are the steps of EDA'):
                    st.write(steps_eda())

            function_agent()

            st.subheader('Variable of study')
            user_question_variable = st.text_input('What variable are you interested in')
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
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                    
                    with st.sidebar:
                        with st.expander("The importance of framing a data science problem approriately"):
                            st.caption(data_science_framing())
                    
                    prompt = st.text_area('What is the business problem you would like to solve?')

                    if prompt:                    
                        wiki_research = wiki(prompt)
                        my_data_problem = chains_output(prompt, wiki_research)[0]
                        my_model_selection = chains_output(prompt, wiki_research)[1]
                        with st.sidebar:
                            with st.expander("Is one algorithm enough?"):
                                st.caption(algorithm_selection())

                            
                        st.write(my_data_problem)
                        st.write(my_model_selection)

                        formatted_list = list_to_selectbox(my_model_selection)
                        selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                        if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                            st.subheader("Solution")
                            solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                            st.write(solution)
    with tab2:
        st.header("ChatBox")
        st.write("🤖 Welcome to the AI Assistant ChatBox!") 
        st.write("Got burning questions about your data science problem or need help navigating the intricacies of your project? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

        st.write("")

        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["How can I assist you?"]
        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=apikey)

        if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


        system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know'""")
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

        response_container = st.container()
        textcontainer = st.container()

       
        with textcontainer:
            query = st.text_input("Hello! How can I help you? ", key="input")
            if query:
                with st.spinner("thinking..."):
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    context = find_match(refined_query)
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i],key=str(i))
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                        






