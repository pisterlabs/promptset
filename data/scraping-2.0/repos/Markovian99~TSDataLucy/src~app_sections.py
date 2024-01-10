import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATE_VAR, DATA_FRACTION, APP_NAME, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR
from app_utils import (generate_responses, initialize_session_state, identify_categorical, process_ts_data, 
                       num_tokens_from_string, identify_features_to_analyze, create_knowledge_base, generate_kb_response)



def run_upload_and_settings():
    """This function runs the upload and settings container"""
    general_context = st.session_state["general_context"]
    brief_description = st.text_input("Please provide a brief description of the data file (e.g. This is market data for the S&P500)", "")
    if len(brief_description)>0:
            general_context = general_context + "The following brief description of the data was provided: "+ brief_description + "\n"
            st.session_state["general_context"] = general_context

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        #copy the file to "raw" folder
        with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

        dataframe = pd.read_csv(uploaded_file)

        # get the date feature
        all_features=dataframe.columns.tolist()
        date_index = all_features.index(st.session_state["date_column"])
        date_column = st.selectbox('What is the primary date column?', all_features, date_index)
        st.session_state["date_column"]=date_column

        try:        
            dataframe[st.session_state["date_column"]] = pd.to_datetime(dataframe[st.session_state["date_column"]], format="%Y-%m-%d")
        except Exception as e:
            print(e)
            st.warning("Could not convert date column to datetime format. Please select a different date column.")

        if st.session_state["uploaded_file"] != uploaded_file.name:
            st.session_state["categorical_features"]=["None"]+identify_categorical(dataframe)
            sum_cols, mean_cols=identify_features_to_analyze(dataframe)
            st.session_state["numeric_features"]=sum_cols+mean_cols
            try:
                # use genai to create the default list of features to analyze
                sum_cols, mean_cols=identify_features_to_analyze(dataframe,use_llm=True,prompt_prefix=general_context)
                st.session_state["features_to_sum"]=sum_cols
                st.session_state["features_to_mean"]=mean_cols
                print("Features to analyze: ")
                print(sum_cols+mean_cols)
            except Exception as e:
                print(e)
                st.session_state["features_to_sum"]=st.session_state["numeric_features"]
                st.session_state["features_to_mean"]=st.session_state["numeric_features"]

        st.session_state["uploaded_file"] = uploaded_file.name
        st.session_state["start_date"]=dataframe[st.session_state["date_column"]].min()
        st.session_state["end_date"]=dataframe[st.session_state["date_column"]].max()
        st.write(dataframe)

    if uploaded_file is not None:    
        col1, col2 = st.columns(2)
        with col1:
            d_min = st.date_input("Analysis Start Date", value=st.session_state["start_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])
            st.session_state["d_min"]=d_min
        with col2:
            d_max = st.date_input("Analysis End Date", value=st.session_state["end_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])
            st.session_state["d_max"]=d_max

        selected_features_sum = st.multiselect('What are the features to analyze and aggregate by summing?', st.session_state["numeric_features"], default=st.session_state["features_to_sum"])
        st.session_state["selected_features_sum"]=selected_features_sum
        selected_features_mean = st.multiselect('What are the features to analyze and aggregate by taking the mean?', st.session_state["numeric_features"], default=st.session_state["features_to_mean"])
        st.session_state["selected_features_mean"]=selected_features_mean

        #check if mean and sum features overlap
        if len(set(selected_features_sum).intersection(set(selected_features_mean)))>0:
            st.warning("The features to sum and mean overlap. Please select different features.")
        

        # streamlit header    
        by_var = st.selectbox(f"(Optional) Select Group By Variable ", st.session_state["categorical_features"])
        st.session_state["by_var"]=by_var


def run_report_gererator():
    template=""
    general_context = st.session_state["general_context"]
    model = st.session_state["generation_model"]
    # if by_var is not set, set it to None
    if st.session_state["by_var"] =="None":
        by_var = None
    else:
        by_var = st.session_state["by_var"]

    st.header("Summary Analysis Reports")
    col1, col2 = st.columns(2)
    with col1:
        # checkbox for each type of report
        field_summary = st.checkbox("Field Descriptions", value=True)
        data_summary = st.checkbox("Data Summary", value=True)
    with col2:
        # checkbox for each type of report
        recent_summary = st.checkbox("Recent Data Analysis", value=True)
        trend_summary = st.checkbox("Trend Summary", value=False)

    if by_var:
        st.header("Group By Analysis Reports")
        col1, col2 = st.columns(2)
        with col1:
            # checkbox for each type of report
            data_summary_by_group = st.checkbox("Data Summary by Group", value=True)
            compare_by_group = st.checkbox("Compare Data by Group", value=False)            
        with col2:
            # checkbox for each type of report
            recent_summary_by_group = st.checkbox("Recent Data Analysis by Group", value=False)
            trend_summary_by_group = st.checkbox("Trend Summary by Group", value=False)
    # User requested report
    requested_report = st.text_input("(Optional) Enter prompt for ad-hoc report", "")

    # Generate button
    generate_button = st.button("Generate Responses")

    if generate_button:
        dataframe = pd.read_csv(os.path.join("../data/raw/"+st.session_state["uploaded_file"]))
        dataframe[st.session_state["date_column"]] = pd.to_datetime(dataframe[st.session_state["date_column"]])
        #subset dataframe based on min and max dates
        dataframe = dataframe[(dataframe[st.session_state["date_column"]].dt.date>=st.session_state["d_min"]) & (dataframe[st.session_state["date_column"]].dt.date<=st.session_state["d_max"])]

        drop_features = [f for f in st.session_state["numeric_features"] if (f not in  st.session_state["selected_features_sum"]+ st.session_state["selected_features_mean"]) and f!=st.session_state["date_column"] and f!=by_var]
        dataframe = dataframe.drop(drop_features, axis=1)

        # process time series data to save descriptive information for prompts
        process_ts_data(dataframe, st.session_state["date_column"], by_var)

        # Open the files in read mode into Python dictionary then back to a JSON string
        with open(PROCESSED_DOCUMENTS_DIR+'head.txt', 'r') as f:
            head_str = f.read()
        with open(PROCESSED_DOCUMENTS_DIR+'summary_all.txt', 'r') as f:
            summary_all_str = f.read()
        with open(PROCESSED_DOCUMENTS_DIR+'summary.txt', 'r') as f:
            summary_str = f.read()
        with open(PROCESSED_DOCUMENTS_DIR+'start.txt', 'r') as f:
            start_str = f.read()
        with open(PROCESSED_DOCUMENTS_DIR+'recent.txt', 'r') as f:
            recent_str = f.read()

        prompt_context= general_context + "\n This is an example of the first set of rows \n"+head_str +"\n"+"Please describe what the data fields may represent."
        #if checked, try to produce a field summary
        if field_summary:
            field_summary_response = generate_responses(prompt_context, model, template)
            st.header(f"Field Summary")
            st.write(field_summary_response)
        
        if data_summary:
            prompt_context = general_context + "Please summarize the data provided and consider this json string summarizing the data: \n"+ summary_all_str
            data_summary_response = generate_responses(prompt_context, model, template)
            st.header(f"Data Summary")
            st.write(data_summary_response)
        if recent_summary:
            prompt_context = general_context + "Compare the aggregated data from the start period with the most recent period to provide analysis of the most recent period.\n Start period:\n"+ start_str+"\n Recent period:\n"+recent_str
            recent_summary_response = generate_responses(prompt_context, model, template)
            st.header(f"Recent Data Analysis")
            st.write(recent_summary_response)

        if by_var:
            if data_summary_by_group:
                # read in the summary data into a string
                with open(PROCESSED_DOCUMENTS_DIR+'comparison.txt', 'r') as f:
                    group_summaries = f.read()           
                prompt_context = general_context + group_summaries + "Please summarize the data provided by group."
                data_summary_response = generate_responses(prompt_context, model, template)
                st.header(f"Data Summary by {by_var}")
                st.write(data_summary_response)
            if compare_by_group:
                # read in the comparison data into a string
                with open(PROCESSED_DOCUMENTS_DIR+'comparison.txt', 'r') as f:
                    group_summaries = f.read()        
                prompt_context = general_context + group_summaries + "Please compare the metrics from the different sub-groups to each other."
                comparison_response = generate_responses(prompt_context, model, template)
                st.header(f"{by_var} Comparison Analysis")
                st.write(comparison_response)
            if recent_summary_by_group:
                # read in the data into a strings
                with open(PROCESSED_DOCUMENTS_DIR+'start_by_group.txt', 'r') as f:
                    json_start_by_group = f.read() 
                with open(PROCESSED_DOCUMENTS_DIR+'/recent_by_group.txt', 'r') as f:
                    json_recent_by_group = f.read()     
                prompt_context = general_context + f"Compare the data from the start period with the most recent period for each {by_var} to provide analysis of the most recent period.\n Start period:\n"+ \
                                json_start_by_group+"\n Recent period:\n"+json_recent_by_group
                recent_summary_response = generate_responses(prompt_context, model, template)
                st.header(f"Recent Data Analysis by {by_var}")
                st.write(recent_summary_response)   

        prompt_context=""
        if len(requested_report)>1:
            print(f"Length of prompt: {len(requested_report)}")
            # Print the JSON string
            prompt_requested = requested_report + prompt_context + "\n" + requested_report
            requested_response = generate_responses(prompt_requested, model, template)
            st.header(requested_report)
            st.write(requested_response)


def run_chatbot():
    template=""
    general_context = st.session_state["general_context"]
    model = st.session_state["generation_model"]
    # if by_var is not set, set it to None
    if st.session_state["by_var"] =="None":
        by_var = None
    else:
        by_var = st.session_state["by_var"]

    # Start button
    start_button = st.button("Start Chatting")

    if start_button:
        dataframe = pd.read_csv(os.path.join("../data/raw/"+st.session_state["uploaded_file"]))
        dataframe[st.session_state["date_column"]] = pd.to_datetime(dataframe[st.session_state["date_column"]])
        #subset dataframe based on min and max dates
        dataframe = dataframe[(dataframe[st.session_state["date_column"]].dt.date>=st.session_state["d_min"]) & (dataframe[st.session_state["date_column"]].dt.date<=st.session_state["d_max"])]

        drop_features = [f for f in st.session_state["numeric_features"] if (f not in  st.session_state["selected_features_sum"]+ st.session_state["selected_features_mean"]) and f!=st.session_state["date_column"]]
        dataframe = dataframe.drop(drop_features, axis=1)

        # process time series data to save to knowledge base
        create_knowledge_base(dataframe, st.session_state["date_column"], by_var)
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input("What are the fields in my data?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):                
                response = generate_kb_response(prompt, model, template) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

        # if st.session_state["generated_responses"] and not st.session_state["cleared_responses"]:
        #     clear_button = st.button("Clear Responses")

        # if clear_button and not st.session_state["cleared_responses"]:        
        #     print(st.session_state["responses"])
        #     st.session_state["generated_responses"]=False
        #     st.session_state["responses"] = []
        #     st.session_state["cleared_responses"]=True

        # elif clear_button:
        #     st.write("No responses to clear - please generate responses")
        #         # responses = []
        #         # ratings = [None, None, None]

        #llm = HuggingFacePipeline(pipeline=pipeline)