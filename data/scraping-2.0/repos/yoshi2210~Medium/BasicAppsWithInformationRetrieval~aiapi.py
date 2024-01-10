# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import config
import tiktoken
import os
import json
from azure.cosmos import CosmosClient, PartitionKey
import datetime
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

openai.api_key = config.DevelopmentConfig.OPENAI_KEY
os.environ['OPENAI_API_KEY'] = config.DevelopmentConfig.OPENAI_KEY

def generateChatResponse(prompt):
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)

    # https://platform.openai.com/docs/guides/chat/introduction
    try:
        answer = response['choices'][0]['message']['content'].replace('\n','<br>') # as html
    except:
        answer = 'OMGGGGG, cannot get response from API'

    return answer

history_filepath = os.path.join(os.path.dirname(__file__), 'cosmosdb_chathistory.json')
# history_filepath = 'chathistory.json'

def generateChatResponseGivenHistory(prompt):
    print(history_filepath)

    # history check
    if os.path.exists(history_filepath):
        with open(history_filepath, 'r') as f:
            try:
                # load previous history data
                history_json_data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print("Error loading JSON:", e)
                history_json_data = {}
        messages = []
        # including history into messages

        # if history is long, just select 3 
        if len(history_json_data)>8:
            short_term_memory_data = history_json_data[-6:] # only 3 conversations
        else:
            short_term_memory_data = history_json_data

        # add history message as assistant to message, as list from json load data are stored as []
        for memory_thread in short_term_memory_data:
            messages.append(memory_thread)

    # if no history
    else:
        history_json_data={}
        short_term_memory_data = []
        messages = []
        
    # Any messages must starts with role system
    messages.append({"role": "system", "content": f"You are a helpful assistant"})

    # Add original prompt
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    print(messages)

    # Ask GPT
    # https://platform.openai.com/docs/guides/chat/introduction
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    try:
        answer = response['choices'][0]['message']['content'].replace('\n','<br>') # as html
    except:
        answer = 'OMGGGGG, cannot get response from API'

    # save chat history to history_json_data

    ##
    userPK = {}
    userPK['userid'] = 'AAAA@BBB.com'

    ## original user prompt 
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    ## LLM answer for this case
    llmanswer = {}
    llmanswer['role'] = 'assistant'
    llmanswer['content'] = answer


    history_json_data.append(question)
    history_json_data.append(llmanswer)

    history_json_data["timestamp"] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    if "conversation" in history_json_data:
        history_json_data["conversation"] = [history_json_data["conversation"], question]
    else:
        history_json_data["conversation"] = question


    with open(history_filepath, 'w') as f:
        try:
            json.dump(history_json_data, f, indent=4)
        except json.decoder.JSONDecodeError as e:
            print("Error saving JSON:", e)

    return answer


def generateChatResponseGivenHistoryCosmosDB(prompt,email):
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print('prompt is -----'+prompt)
    print('email is ------'+email)
    # 接続情報を設定する
    ENDPOINT = config.DevelopmentConfig.COSMOS_ENDPOINT 
    KEY = config.DevelopmentConfig.COSMOS_KEY 

    # Cosmos DB クライアントを初期化する
    client = CosmosClient(url=ENDPOINT, credential=KEY)

    # データベースを定義する, AzurePortal側でつくったなぜかContainerだけがエラーではじかれるから
    DATABASE_NAME  = 'ChatHistory'
    CONTAINER_NAME  = 'container1'

    # AzureAD b2cの情報たよりに履歴引っ張る
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # 最近のチャット３つを履歴でとっておくか雑だけど
    QUERY = "SELECT TOP 4 c.conversation, c.timestamp FROM ChatLogs c WHERE c.email = @email ORDER BY c.timestamp DESC"
    params = [
        dict(name='@email', value=email)
    ]
    items = container.query_items(
        query=QUERY, parameters=params, enable_cross_partition_query=True
    )

    # GPTに投げるメッセを準備する
    messages = []  
    messages.append({"role": "system", "content": f"You are a helpful assistant"})

    for item in items:
        chat_history = item['conversation'][0] # user input
        messages.append(chat_history)
        chat_history = item['conversation'][1] # content, that is, previous answer
        messages.append(chat_history)      
        # print(item['conversation'][0])

    # Add original user prompt
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    print('-input message-'*10)
    print(messages)
    print('-'*50)
    
    # Ask GPT
    # https://platform.openai.com/docs/guides/chat/introduction
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    try:
        answer = response['choices'][0]['message']['content'].replace('\n','<br>') # as html
    except:
        answer = 'OMGGGGG, cannot get response from API'

    print('-'*50)
    print(answer)
    print('-'*50)

    ## LLM answer for this case
    GPTanswer = {}
    GPTanswer['role'] = 'assistant'
    GPTanswer['content'] = answer
    print(GPTanswer)

    # save chat history to history_dict_data
    history_dict_data = {}
    history_dict_data["id"] = str(email)+ datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    history_dict_data["email"] = email
    history_dict_data["timestamp"] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    history_dict_data["conversation"] = question
    history_dict_data["conversation"] = [history_dict_data["conversation"],GPTanswer]
    history_dict_data["feedback"] = 1
    history_dict_data["type"] = 'CHAT'

    print(history_dict_data)

    try:
        container.create_item(history_dict_data)
    except json.decoder.JSONDecodeError as e:
            print("Error saving DATA to CosmosDB:", e)    

    return answer


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS #Chroma

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Just load vector store data    
vdb_faiss_path = os.path.join(os.path.dirname(__file__), 'vectordb-faiss')
# history_filepath = 'chathistory.json'
print(f"vdb_faiss_path={vdb_faiss_path}")
vector_store = FAISS.load_local(vdb_faiss_path,OpenAIEmbeddings())



def generateChatResponseQnACosmosDB(prompt,email):
    print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

    # 接続情報を設定する
 
    ENDPOINT = config.DevelopmentConfig.COSMOS_ENDPOINT 
    KEY = config.DevelopmentConfig.COSMOS_KEY 

    # Cosmos DB クライアントを初期化する
    client = CosmosClient(url=ENDPOINT, credential=KEY)

    # データベースを定義する, AzurePortal側でつくったなぜかContainerだけがエラーではじかれるから
    DATABASE_NAME  = 'ChatHistory'
    CONTAINER_NAME  = 'container1'


    # AzureAD b2cの情報たよりに履歴引っ張る
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # # Just load vector store data    
    # vdb_faiss_path = os.path.join(os.path.dirname(__file__), 'vectordb-faiss')
    # # history_filepath = 'chathistory.json'
    # print(f"vdb_faiss_path={vdb_faiss_path}")
    # vector_store = FAISS.load_local(vdb_faiss_path,OpenAIEmbeddings())

    # First: Simple version
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # pdf_qa = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever(), return_source_documents=True)

    # Second:A little bit complicated version
    system_template="""Use the following pieces of context to answer the users question.
    Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    All answer must be in Japanese.
    ----------------
    {summaries}"""

    messages_langchain = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    chian_prompt = ChatPromptTemplate.from_messages(messages_langchain)

    chain_type_kwargs = {"prompt": chian_prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1024)  # Modify model_name if you have access to GPT-4
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    

    # Add original user prompt
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    # messages.append(question)
    # print(messages)
    

    # Simple version
    # chat_history = []
    #result = pdf_qa({"question": prompt, "chat_history": chat_history})
    # Second version
    result = chain(prompt)

    # how to answer as LLM 
    def print_result(result):
        output_text = f"""### Question: 
        {prompt}
        ### Answer: 
        {result['answer']}
        ### All relevant sources:
        {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
        """
        # once we put reference sentence into history too, token limitation happens with ease. 
        # output_text = f"""### Question: 
        # {prompt}
        # ### Answer: 
        # {result['answer']}
        # ### Sources: 
        # {result['sources']}
        # ### All relevant content:
        # {' '.join(list(set([doc.page_content for doc in result['source_documents']])))}
        # ### All relevant sources:
        # {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
        # """
        return output_text


    answer = print_result(result)

    print('-'*50)
    print(answer)
    print('-'*50)

    ## LLM answer for this case
    GPTanswer = {}
    GPTanswer['role'] = 'assistant'
    GPTanswer['content'] = answer
    print(GPTanswer)

    # save chat history to history_dict_data
    history_dict_data = {}
    history_dict_data["id"] = str(email)+ datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    history_dict_data["email"] = email
    history_dict_data["timestamp"] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    history_dict_data["conversation"] = question
    history_dict_data["conversation"] = [history_dict_data["conversation"],GPTanswer]
    history_dict_data["feedback"] = 1
    history_dict_data["type"] = 'CHAT'

    print(history_dict_data)

    try:
        container.create_item(history_dict_data)
    except json.decoder.JSONDecodeError as e:
            print("Error saving DATA to CosmosDB:", e)    

    return answer.replace('\n','<br>') # as html



# COSMOS DB model
# new_item = {
#     "user_id": "A00000001",
#     "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
#     "conversation": [
#         {"role": "user", "content": history[-1]["user"]},
#         {"role": "assistant", "content": completion.choices[0]['message']['content']},
#     "feedback": 1
#     ]
# }


## CHAT COMPLETION Note
# openai.ChatCompl  etion.create(
#     model="gpt-3.5-turbo",
#     messages=messages)
#     # messages=[
#             # {"role": "system", "content": "You are a helpful assistant."},
#             # {"role": "user", "content": "Who won the world series in 2020?"},
#             # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#             # {"role": "user", "content": "Where was it played?"}
#     #]
