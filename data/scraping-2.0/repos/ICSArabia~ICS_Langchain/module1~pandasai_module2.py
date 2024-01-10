from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain import LLMMathChain
from langchain.agents import Tool
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from pandasai.llm import OpenAI
from pandasai import SmartDataframe

import sqlite3
import pandas as pd
import os
import base64
import boto3
import pytz
import argparse
import constants

base_64_image = None
global base64_image

global plot_var
plot_var = None


os.environ["OPENAI_API_KEY"] = constants.APIKEY

def get_base64_of_png_image(png_image_path):
  
  """Returns the base64 of a PNG image.

  Args:
    png_image_path: The path to the PNG image.

  Returns:
    A base64 string of the PNG image.
  """

  with open(png_image_path, "rb") as f:
    image_bytes = f.read()

  base64_image = base64.b64encode(image_bytes).decode("utf-8")

  return base64_image

def get_base64_of_latest_png_file_based_on_date_time(directory_path):
  
  """Returns the base64 of the latest PNG file based on date time in a given directory.

  Args:
    directory_path: The path to the directory containing the PNG files.

  Returns:
    A base64 string of the latest PNG file based on date time, or None if no PNG files are found in the directory.
  """

  # Get a list of all the PNG files in the directory.
  png_files = []
  for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
      png_files.append(filename)

  # Sort the PNG files by creation date, newest first.
  png_files.sort(key=lambda filename: os.path.getctime(os.path.join(directory_path, filename)), reverse=True)

  # Get the path to the latest PNG file.
  if len(png_files) >= 0:
    latest_png_file_path = os.path.join(directory_path, png_files[0])

  # Get the base64 of the PNG file.
  base64_image = get_base64_of_png_image(latest_png_file_path)

  return base64_image

def read_dataframe_from_db(table_name, prefix):
    
    # db_name = f"{prefix}.db"
    db_name = os.path.join(os.getcwd(), 'module1', f"{prefix}.db")
    print('db_name accessed',db_name)
    connection = sqlite3.connect(db_name)
    
    # Create a SQL query to select all of the data from the database table.
    sql_query = f"SELECT * FROM '{table_name}'"

    print(db_name)
    print(sql_query)
    print(connection)
    

    # Read the data from the database table into a DataFrame.
    df = pd.read_sql_query(sql_query, connection)
    
    #pre-process the df before storing
    df.columns=  df.columns.str.strip()
    # df.drop('Unnamed: 0', axis=1, inplace=True)

    #setting header row
    df.columns = df.iloc[0]

    # Drop the second row, as it is now the column names
    df = df[1:]
    df = df.fillna(0)

    # Close the database connection.
    connection.close()

    # Return the DataFrame.
    return df


def db_dataframe_agent(df):
    chat = ChatOpenAI(model_name= 'gpt-4-1106-preview', temperature = 0.0)
    agent = create_pandas_dataframe_agent(chat,df,verbose= True,handle_parsing_errors=True)
    return agent

def read_persisted_vextorstore_db(prefix):
    # defining pre-persisted dir prefix of vecotorstore to load
    persist_directory = 'db'+ '_' +prefix

    persist_directory  = os.path.normpath(os.path.join(os.getcwd(), persist_directory))

    print(persist_directory)

    # loaded pre-persisted vectordb
    vectordb = Chroma(persist_directory= persist_directory, embedding_function=OpenAIEmbeddings())
    
    return vectordb

def retriever_vextorstore_db(vectordb,source_file_path):

    # source_file_path = 'D:\ics-arabia\chat_assistant_8\hr\Final DHP Enterprise Annual Financial Report  file 3_.pdf'
    # source_file_path = None
    print('source file path:-', source_file_path)
    
    # retrieve indexes of vectoredb to use
    if source_file_path == None:
        retriever = vectordb.as_retriever(search_kwargs={"k":4})
    else:
        #with filter to search / lookup in specfic documents only
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5})
    return retriever


def qa_chain_with_memory_and_search(retriever):

    # create the model
    llm = ChatOpenAI(model='gpt-4',
                        temperature=0.0,)


    # create the chain to answer questions 

    global qa_chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True   #<---- for me to test / see all output on term
                                    )

    # memory buffer
    memory = ConversationBufferMemory(memory_key="chat_history", k=5, return_messages=True)

    # internet search
    search = DuckDuckGoSearchRun()
    
    return llm,qa_chain,memory,search

def agent_2(llm,qa_chain,memory,search, user_query,prefix,table_name):

    #Define tool(s)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Retriever",
            func=qa_chain,
            description="retrieves information from the Chroma DB for source questions"
        ),

        Tool(
            name ="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
        
        Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
        ),
        
  
    ]
    
    df = read_dataframe_from_db(table_name,prefix)
    agent_chain = db_dataframe_agent(df)
    
    plot_var = None
    
    return agent_chain.run(user_query)

def upload_directory_to_s3_with_delta(local_directory, s3_bucket_name, s3_subdirectory):
    """Uploads new or modified files in a local directory to a specific subdirectory in an S3 bucket.

    Args:
        local_directory: The path to the local directory to upload.
        s3_bucket_name: The name of the S3 bucket.
        s3_subdirectory: The subdirectory path within the S3 bucket.

    Returns:
        None.
    """

    s3 = boto3.client('s3')
    utc = pytz.UTC

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)

            # Determine the relative path to the file within the local directory
            relative_path = os.path.relpath(local_file_path, local_directory)

            # Construct the S3 key with the subdirectory and relative path
            s3_key = os.path.join(s3_subdirectory, relative_path).replace("\\", "/")

            s3_object = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_key)

            # Check if the file exists in S3 and compare timestamps
            if 'Contents' in s3_object:
                s3_last_modified = s3_object['Contents'][0]['LastModified']
                local_last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(local_file_path))
                local_last_modified = local_last_modified.replace(tzinfo=utc)

                if s3_last_modified >= local_last_modified:
                    continue  # Skip uploading if S3 version is newer or equal

            # Upload the local file to S3 with the subdirectory path
            s3.upload_file(local_file_path, s3_bucket_name, s3_key)
            print(s3_key, "'s upload successful")

def agent_plot(llm,qa_chain,memory,search, user_query,table_name,prefix):

  
    #read data first
    print('table name:-',table_name)
    df = read_dataframe_from_db(table_name,prefix)
    
    message = plot(df,user_query)
    
    
    # agent.chat(user_query)


    # saving the chart to s3 cloud directory


    # to save plots in s3 bucket
    # local_directory_to_upload = 'D:\\ics-arabia\\chat_assistant_8\\charts'
    local_directory_to_upload = os.path.normpath(os.getcwd() + '/charts')
    s3_bucket_name = 'ics-principal-bucket'
    s3_subdirectory = 'charts'  # Specify the target directory within the S3 bucket

    # upload_directory_to_s3_with_delta(local_directory_to_upload, s3_bucket_name, s3_directory)
    upload_directory_to_s3_with_delta(local_directory_to_upload, s3_bucket_name, s3_subdirectory)


    return message


#################################################################################################



def plot(df, user_query):
    global plot_var
    plot_var =1
    print('plot_var2:-----',plot_var)
    import matplotlib
    matplotlib.use('agg')
    print('going into the plotting function')
    print(df.head())
    llm = OpenAI(model='gpt-4',temperature=0.0)
    
    
    user_defined_path = os.path.normpath(os.getcwd() + str('\charts'))
    # check if the directory exists
    if not os.path.exists(user_defined_path):
        os.makedirs(user_defined_path)
    # user_defined_path = '\charts'

    print('Path to charts directory',user_defined_path)

    # sdf = SmartDataframe(df, config={"llm": llm, "save_charts": True,"save_charts_path": user_defined_path})
    sdf = SmartDataframe(df, config={"llm": llm, "save_charts": True,"save_charts_path": user_defined_path, "enable_cache": False})
    # sdf = SmartDataframe(df, config={"llm": llm, "save_charts": True, "enable_cache": False})


    sdf.chat(user_query)

    # sdf = SmartDataframe(df, config={"llm": llm})
    # print('sdf:-',sdf)


    #move from  working directory to specified charts directory
    global base64_image
    message, base64_image = move_png_to_charts(os.getcwd())

    # print(base64_image)

    return message


##############################################################################

def return_base64():
    return base64_image


def return_plot_var():
    return plot_var
###############################################################################

import datetime

def move_png_to_charts(working_directory):
#   """Moves all PNG files in the working directory to a folder named charts,
#   and rename the files by adding the current date and time at the start.

#   Args:
#     working_directory: The working directory.
#   """

#   charts_directory = os.path.join(working_directory, "charts")
#   if not os.path.exists(charts_directory):
#     os.makedirs(charts_directory)

#   for filename in os.listdir(working_directory):
#     if filename.endswith(".png"):
#       now = datetime.datetime.now()
#       new_filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
#       os.rename(os.path.join(working_directory, filename),
#                 os.path.join(charts_directory, new_filename))


    directory_path = os.path.normpath(os.getcwd() + str('\charts'))

    # directory_path = os.getcwd()
    global base64_image
    base64_image = get_base64_of_latest_png_file_based_on_date_time(directory_path) 
    # print(base64_image)

    message = 'Graph Plotted, please click "Visualize Graph" button below to view...'

    # return sdf.chat(processed_user_query)
    return message, base64_image


def router_5(prefix,user_query, llm,qa_chain,memory,search, structured_retriever):
    print('calling router:--- 5')
    llm = ChatOpenAI(model='gpt-4-1106-preview',temperature=0.4)


    # Set up the routing logic
    #First prompt
    prompt = PromptTemplate.from_template("""Only if the question explicitly uses word 'structured table' or word 'data frame', return the exact table name or dataframe name. In the response
                                            just the table name itself is required, without any full stops or appending any extra string. Do not send a string beyond the table name. 
                                            For example if the reponse coming out is 'The table name in the question is 'FA-2'', then only return 'FA-2'
                                            Another example: What are the details of Table 2: Summary of Management Assurances 13 in this case table wont be returned but just 'None'
                                            Another example: What are the details of 'structured table' Summary of Management Assurances 13. In this case the table's name 'Summary of Management Assurances 13' would be returned
                                            If the user asks something like: Who is the Agency Head Defense Health Program in 2021?, this is not to be used for invoking the table name return None
                                            so return none is all secnarios when 'structured table' word is not used
                                             Question: {question}""")

    # router_chain = prompt | ChatOpenAI() | parser
    router_chain = prompt | ChatOpenAI() | StrOutputParser()
    
    # Set up the base chain
    llm_chain = PromptTemplate.from_template("""Respond to the question: Question: {input}""") | ChatOpenAI() | StrOutputParser()

    # Add the routing logic - use the action key to route to appropriate agent
    def select_table(output):
        # table_name = output["table_name"] if output["action"] == "plot" else None
        if output["action"] != None:
            table_name = output['action']
            print('table name only at router 3 stage:-', table_name)
            # router(prefix,user_query, llm,qa_chain,memory,search, table_name)
        else:
            table_name = None
            # router(prefix,user_query, llm,qa_chain,memory,search, table_name)
    
    
    
    # Set up the routing logic
    # second prompt
    prompt2 = PromptTemplate.from_template("""If the question uses words like describe, show, print, details or proceeds to asks about data and finding it from some row-column like condition for retrieving data, structured data from a csv, excel file, df, respond with `data`
                                            Example: describe the table details then 'data' would be returned
                                            Another example: What are the details of Table 13'. In this case the 'data' would be returned
                                            If the condition is about plotting, charting, drawing or producing an image than respond with 'plot'. 
                                            If it is unclear in all other conditions also return 'data'
                                            Question: {question}""")

    router_chain_2 = prompt2 | ChatOpenAI() | StrOutputParser()
    
    
    

    # Set up the base chain
    llm_chain = PromptTemplate.from_template("""Respond to the question: Question: {input}""") | ChatOpenAI() | StrOutputParser()

    # Add the routing logic - use the action key to route to appropriate agent
    def select_chain(output):
        # table name
        ## table name to come from the drop down selection
        table_name = structured_retriever
        print('table name parsed from user query:--',table_name)
            
        if output["action2"] == "data":
            print('agent_2 for getting data')
            return agent_2(llm,qa_chain,memory,search, user_query,prefix, table_name)
            # return agent_1(llm,qa_chain,memory,search, user_query)
        
        elif output["action2"] == 'plot':
            print('agent_plot for plotting')
            # table_name = output["action"]
            return agent_plot(llm,qa_chain,memory,search, user_query, table_name, prefix)
        else:
            raise ValueError

    # Create the final chain
    # Generate the action, and pass that into `select_chain`
    chain = RunnableMap({
        "action": router_chain,
        "action2": router_chain_2,
        "input": lambda x: x["question"]
    }) | select_chain

    
    return chain.invoke({"question":user_query})

argparser = argparse.ArgumentParser()
argparser.add_argument("--query", help="Question to ask")
argparser.add_argument("--path", help="Path to the document")

args = argparser.parse_args()

user_query = args.query
structured_retriever=args.path

prefix = 'Structured'
vectordb = read_persisted_vextorstore_db(prefix)



# retrieving vectors of splitted-text-of-langchian-docs from vectorstore persisted db
retriever= retriever_vextorstore_db(vectordb,source_file_path=None)


#initalize qa_chain function / helper function for agent
llm,qa_chain,memory,search = qa_chain_with_memory_and_search(retriever)

result = router_5(prefix,user_query, llm,qa_chain,memory,search, structured_retriever)

plot_var = return_plot_var()
print('plot_var:-----',plot_var)
if plot_var == 1:
    base64_image = return_base64()
    print('base64_image', base64_image)

# answer, source = break_response_source(result)
print('answer:--',result)
# print('source:--', source)