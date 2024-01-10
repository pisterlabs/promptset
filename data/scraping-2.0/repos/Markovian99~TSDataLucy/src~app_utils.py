import pandas as pd
import streamlit as st
import os
import json
import tiktoken
import numpy as np
import time

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
	MarkdownTextSplitter,
	PythonCodeTextSplitter,
	RecursiveCharacterTextSplitter)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAIChat

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATA_FRACTION, EMBEDDING_MODELS, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR, DATE_VAR

from bardapi import Bard
import openai


# make sure load_dotenv is run from main app file first
openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

#bard = Bard(token=os.getenv('BARD_API_KEY'))


def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "date_column": DATE_VAR,
        "categorical_features": [],
        "numeric_features": [],
        "features_to_sum": [],      
        "features_to_mean": [],  
        "selected_features_sum": [],  
        "selected_features_mean": [],
        "start_date": None,
        "end_date": None,
        "d_min": None,
        "d_max": None,
        "uploaded_file": None,
        "by_var" : "None",
        "generation_model": MODELS[0],
        "general_context": "",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": []
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def identify_categorical(df, unique_threshold=100, max_portion=0.1):
    """Identify categorical columns in a dataframe."""
    categorical_cols = []
    max_unique = min(int(len(df) * max_portion), unique_threshold)
    
    potential_categorical_cols = df.select_dtypes(include=['int64', 'object']).columns
    for col in potential_categorical_cols:
        if df[col].nunique() <= max_unique and df[col].nunique() > 1:
            categorical_cols.append(col)
            
    return categorical_cols

def identify_features_to_analyze(df, unique_min=0, use_llm=False, prompt_prefix=""):
    """Identify possible features_to_analyze in a dataframe."""
    cols = []
    sum_cols = []
    mean_cols = []

    # select all numeric columns
    numeric_cols =  df.select_dtypes('number').columns.to_list()
    for col in numeric_cols:
        if df[col].nunique() > unique_min:
            if df[col].max()*df[col].min() < 0:
                cols.append(col)
                sum_cols.append(col)
            else:
                cols.append(col)
                mean_cols.append(col)

    if len(cols)==0:
        st.warning("No numeric columns appropriate for analyzing were found in data.")
    elif use_llm and st.session_state["generation_model"]!="":
        model = st.session_state["generation_model"]
        template=""
        prompt = prompt_prefix+" The following is a summary of the numeric data in the dataframe: \n\n"+str(df[cols].describe().to_json())+ \
                "\n\nWhat are features that should be analyzed and aggregated by summing? What are features that should be analyzed and aggregated by taking the mean? Please do not mention any features that should not be analyzed and respond in the following format: \n\n" + \
                "sum: feature1, feature2, feature3 \nmean: feature4, feature5, feature6"
        response = generate_responses(prompt, model, template, temperature=0)

        print(response)
        
        # check which columns are in the response which is a string
        sum_cols = [col for col in cols if col.lower() in response.lower()[:response.index("mean:")]]
        mean_cols = [col for col in cols if col.lower() in response.lower()[response.index("mean:"):]]

    return sum_cols, mean_cols

def mode(x):
    # Check for missing values and empty groups
    if x.isnull().all() or x.empty:
        return np.nan
    else:
        # Use value_counts and return the index of the first occurrence of the maximum value
        mode = x.value_counts().idxmax()
        return mode

def missing(x):
    return x.isnull().sum()

# Specify the aggregation functions for each column
#agg_funcs = {col: [('mode', mode), ('missing', missing)] for col in object_cols}
# Apply groupby and agg
#df_obj = df[object_cols+grouping].groupby(grouping).agg(agg_funcs)


def group_as_needed(df_input, grouping):
    """Group data if needed."""
    df = df_input.copy()
    # only keep numeric and object columns not entirely missing
    numeric_cols =  df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.to_list()    
    numeric_cols = [col for col in numeric_cols if df[col].isnull().sum()<len(df[col]) and col not in grouping]
    object_cols = df.select_dtypes(include=['object']).columns.to_list()
    object_cols = [col for col in object_cols if df[col].isnull().sum()<len(df[col]) and col not in grouping and col!=st.session_state["date_column"]]

    # check which numeric columns to sum and which to mean
    sum_cols = list(set(numeric_cols).intersection(set(st.session_state["selected_features_sum"])))
    mean_cols = list(set(numeric_cols).intersection(set(st.session_state["selected_features_mean"])))

    if len(df[grouping].groupby(grouping).size())==len(df):
        #set index as grouping sorted for json file
        df.set_index(grouping, inplace=True, drop=False)
    else:
        # use pandas groupby to aggregate. For objects use the mode and add postfix "mode", for numeric use the sum or mean
        df_num_sum = df[list(set(sum_cols+grouping))].groupby(grouping).sum()
        df_num_mean = df[list(set(mean_cols+grouping))].groupby(grouping).mean()
        if len(object_cols)>0:
            df_obj = df[list(set(object_cols+grouping))].groupby(grouping).agg(lambda x:x.mode())
            df_obj.columns = [f"{col}_mode" for col in df_obj.columns]
            df = pd.concat([df_num_sum,df_num_mean, df_obj], axis=1)
        else:
            df = pd.concat([df_num_sum,df_num_mean], axis=1)
        df = df.reset_index().set_index(grouping, drop=False)
    return df

def process_data_to_file(df_input:pd.DataFrame, date_var=None, by_var=None, file_name='data.txt', prompt_frac=.45, header="Data:\n", date_format='%Y-%m-%d'):
    """Process data and save to various text files for streamlit app."""
    data = df_input.copy()
    #now that processing is done, save the date as a string
    if date_var!=None:
        data[date_var] = data[date_var].dt.strftime(date_format)
    # if no group requested try to save the data and if too large, summarize and save
    if date_var==None and by_var==None:
        data_json = data.to_json()
        num_tokens = num_tokens_from_string(data_json)
        if  num_tokens > int(prompt_frac*MAX_TOKENS):
            # create a subset of the dateframe to a sample of the by_var
            data = data.describe(include='all')
            data_json = data.to_json()
            # write streamlit warning that the data only summarized
            st.warning(f"Data was summarized to fit within the token limit of {int(prompt_frac*MAX_TOKENS)} tokens.")
        data_json = json.loads(data_json)
    elif date_var==None:
        data_json = group_as_needed(data, grouping=[by_var]).to_json()
        if  num_tokens_from_string(data_json) > int(prompt_frac*MAX_TOKENS):
            # create a subset of the dateframe to a sample of the by_var
            data = df_input.describe(include='all')
            data_json = data.to_json()
            # write streamlit warning that the data was only summarized
            st.warning(f"Data was summarized to fit within the token limit of {int(prompt_frac*MAX_TOKENS)} tokens.")
        data_json = json.loads(data_json)
    elif by_var==None:
        data_json = group_as_needed(data, grouping=[date_var]).to_json()
        if  num_tokens_from_string(data_json) > int(prompt_frac*MAX_TOKENS):
            # create a subset of the dateframe to a sample of the by_var
            data = df_input.describe(include='all')
            data_json = data.to_json()
            # write streamlit warning that the data was only summarized
            st.warning(f"Data was summarized to fit within the token limit of {int(prompt_frac*MAX_TOKENS)} tokens.")
        data_json = json.loads(data_json)
    else:
        data_json = group_as_needed(data, grouping=[date_var,by_var]).to_json()
        if  num_tokens_from_string(data_json) > int(prompt_frac*MAX_TOKENS) and by_var:
            data = group_as_needed(data.reset_index(drop=True), grouping=[by_var])
            data_json = data.to_json()
            num_tokens = num_tokens_from_string(data_json)
            if  num_tokens > int(prompt_frac*MAX_TOKENS):
                # create a subset of the dateframe to a sample of the by_var
                data_json = data[data[by_var].isin(np.random.choice(data[by_var].unique(),size=int(len(data[by_var].unique())*prompt_frac*MAX_TOKENS/num_tokens),replace=False))].to_json()
                # write streamlit warning that the data was subset
                st.warning(f"Data was subset to {len(data)} rows to fit within the token limit of {int(prompt_frac*MAX_TOKENS)} tokens.")
        data_json = json.loads(data_json)
    with open(PROCESSED_DOCUMENTS_DIR+file_name, 'w') as json_file:
        json_file.write(header)
        json.dump(data_json, json_file, indent=4)


# This is a dummy function to simulate generating responses.
def generate_responses(prompt, model, template="", temperature=0):
    response = "No model selected"

    if model != "None":
        st.session_state["generation_models"] = model

        if model.startswith("Google"):
            this_answer = bard.get_answer(prompt)
            response = this_answer['content']
        elif model.startswith("OpenAI: "):
            # try to call openai and if it fails wait 5 seconds and try again
            try:
                response_full = openai.Completion.create( model=model[8:],  messages=[{"role": "user", "content": prompt }], temperature=temperature)
            except:
                st.warning("OpenAI API call failed. Waiting 5 seconds and trying again.")
                time.sleep(5)
                response_full = openai.ChatCompletion.create( model=model[8:],   messages=[{"role": "user", "content": prompt }], temperature=temperature)
            response = response_full['choices'][0]['message']['content']

    return response


def process_ts_data(df_input:pd.DataFrame, date_var='date', by_var=None):
    """Process time series data and save to various json files for streamlit app.
    - summary_all.txt: summary of all columns
    - summary.txt: summary of numeric columns only
    - head.txt: first 7 rows of dataframe
    - start.txt: first 7 days of dataframe
    - recent.txt: last 7 days of dataframe
    - compare.txt: numeric summary comparison by group    
    """
    df = df_input.copy()

    # Summarize the dataframe (all and numeric only)
    data = json.loads(df.describe(include='all').to_json())
    # Now, let's write this data to a file
    with open(PROCESSED_DOCUMENTS_DIR+'summary_all.txt', 'w') as json_file:
        json_file.write("Summary of all columns:\n")
        json.dump(data, json_file, indent=4)

    data = json.loads(df.describe().to_json())
    # Now, let's write this data to a file
    with open(PROCESSED_DOCUMENTS_DIR+'summary.txt', 'w') as json_file:
        json_file.write("Summary of numeric columns:\n")
        json.dump(data, json_file, indent=4)

    use_rows=7
    if len(df.columns)>10:
        use_rows=2
    # keep only the head of dataframe and write json file
    process_data_to_file(df.head(use_rows), date_var=None, by_var=None, file_name='head.txt', prompt_frac=.5*DATA_FRACTION, header=f"First {use_rows} rows of data:\n")

    # keep only the first 7 days in the dataframe and write json file
    data = df[df[date_var] < df[date_var].min() + pd.Timedelta(days=7)].sort_values([date_var])
    process_data_to_file(data, date_var=date_var, by_var=None, file_name='start.txt', prompt_frac=.5*DATA_FRACTION, header=f"First 7 days of data:\n")

    # keep only the last 7 days in the dataframe and write json file
    data = df[df[date_var] > df[date_var].max() - pd.Timedelta(days=7)].sort_values([date_var])
    process_data_to_file(data, date_var=date_var, by_var=None, file_name='recent.txt', prompt_frac=.5*DATA_FRACTION, header=f"Last 7 days of data:\n")

    # if by_var is not None, create a reports by group
    # if we need to only keep a subset of by_var values due to size of data
    if by_var:
        # keep only the first 7 days in the dataframe and write json file
        data = df[df[date_var] < df[date_var].min() + pd.Timedelta(days=7)].sort_values([by_var, date_var])
        process_data_to_file(data, date_var=date_var, by_var=by_var, file_name='start_by_group.txt', prompt_frac=.5*DATA_FRACTION, header=f"First 7 days of data by {by_var}:\n")

        # keep only the last 7 days in the dataframe and write json file
        data = df[df[date_var] > df[date_var].max() - pd.Timedelta(days=7)]  
        process_data_to_file(data, date_var=date_var, by_var=by_var, file_name='recent_by_group.txt', prompt_frac=.5*DATA_FRACTION, header=f"Last 7 days of data by {by_var}:\n")

        group_counts = df[by_var].value_counts()
        group_summaries = ""
        for group_name in group_counts.index:
            group_summaries = group_summaries + str(group_name)+":\n" + df[df[by_var]==group_name].describe().to_json() + "\n\n"
            # count the length of the prompt
            prompt_len = num_tokens_from_string(group_summaries)
            #if prompt is too long, exit for loop
            if prompt_len>int(DATA_FRACTION*MAX_TOKENS):
                st.warning(f"Comparison data may be subset to fit within the token limit.")
                break
        #now let's write the test to a file
        with open(PROCESSED_DOCUMENTS_DIR+'comparison.txt', 'w') as f:
            f.write("Comparison of numeric data by group:\n")
            f.write(group_summaries)


def split_json_doc_with_header(doc):
    """Separate header on line one from json doc and split json dict by keys"""
    try:
        header = doc.page_content.split("\n")[0]
        #print(doc.page_content[len(header)+1:])
        json_dict = json.loads(doc.page_content[len(header)+1:])
        doc_list = []
        for key in json_dict.keys():
            doc_list.append(Document(page_content=header+'Data for ' +str(key)+ ':\n'+str(json_dict[key]), metadata=doc.metadata))
        return doc_list
    except Exception as e:
        print(e)
        print("Unable to split " + doc.metadata['source'])
        return [doc]




def create_knowledge_base(df_input:pd.DataFrame, date_var='date', by_var=None):
    """Create knowledge base for chatbot."""
    process_ts_data(df_input, date_var=date_var, by_var=by_var)
    
    #load all .txt files in processed_documents directory
    loader = DirectoryLoader(f"{PROCESSED_DOCUMENTS_DIR}", glob="*.txt")

    print(f"Loading {PROCESSED_DOCUMENTS_DIR}")
    docs_orig = loader.load()
        
    print(f"Splitting {len(docs_orig)} documents")
    docs = []
    for doc in docs_orig:
        print(doc)
        num_tokens = num_tokens_from_string(doc.page_content)
        if  num_tokens > int(.1*MAX_TOKENS):
            doc_list = split_json_doc_with_header(doc)
            docs.extend(doc_list)
        else:
            docs.append(doc)
    
    print(f"Created {len(docs)} documents")

    # Will download the model the first time it runs
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODELS[0],
        cache_folder="../data/sentencetransformers",
    )
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. You may want to increase the number of CPU's in your noteboook.
        """
    )
    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)  
    # Save the FAISS db 
    db.save_local("../data/faiss-db")

    print(f"FAISS VectorDB has {db.index.ntotal} documents")
    

def generate_kb_response(prompt, model, template=None):

    data_dict = {}
    data_dict['prompt'] = prompt
    data_dict['chat_history'] = []

    if model.startswith("OpenAI: "):
        llm = OpenAIChat(model=model[8:], max_tokens=2000, temperature=TEMPERATURE)
    else:
        return "Please select an OpenAI model."

    # Will download the model the first time it runs
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODELS[0],
        cache_folder="../data/sentencetransformers",
    )
    db = FAISS.load_local("../data/faiss-db", embedding_function)

    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": 6})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True) #, return_source_documents=True

    # prompt_template = """
    # ### System:
    # {context}

    # ### User:
    # {question}

    # ### Assistant:
    # """
    # PROMPT = PromptTemplate(
    #     template=prompt_templatjosleee, input_variables=["context", "question"]
    # )
    # chain_type_kwargs = {"prompt": PROMPT}
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    # query = data_dict['prompt']
    # return qa.run(query)
    response = chain(inputs={'question':data_dict['prompt'], 'chat_history':data_dict['chat_history']})
    for doc in response['source_documents']:
        print(doc)
        print("\n")

    return response['answer']

