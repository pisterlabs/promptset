import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys
import re

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import CSVLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chains import LLMChain

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
opensearch_url = os.environ.get('opensearch_url')
bedrock_region = os.environ.get('bedrock_region')
rag_type = os.environ.get('rag_type')
enableConversationMode = os.environ.get('enableConversationMode', 'enabled')
print('enableConversationMode: ', enableConversationMode)
enableReference = os.environ.get('enableReference', 'false')
enableRAG = os.environ.get('enableRAG', 'true')

# opensearch authorization - id/passwd
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
# opensearch authorization - aws auth
# from requests_aws4auth import AWS4Auth
# credentials = boto3.Session().get_credentials()
# awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

conversationMothod = 'RetrievalQA' # ConversationalRetrievalChain or RetrievalQA

modelId = os.environ.get('model_id')
print('model_id: ', modelId)
isReady = False   
accessType = os.environ.get('accessType')

boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
def get_parameter(modelId):
    if modelId == 'amazon.titan-tg1-large' or modelId == 'amazon.titan-tg1-xlarge': 
        return {
            "maxTokenCount":1024,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        }
    elif modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
        return {
            "max_tokens_to_sample":8191,
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [HUMAN_PROMPT]            
        }
parameters = get_parameter(modelId)

llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    #streaming=True,
    model_kwargs=parameters)

map = dict() # Conversation

# embedding
bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' # amazon.titan-e1t-medium, amazon.titan-embed-g1-text-02 amazon.titan-embed-text-v1
)
    
# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    

        typeArg = str(type(contents))
        print('typeArg: ', typeArg)
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(texts):    
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(texts))
    print('word_kor: ', word_kor)
    
    if word_kor:
        #prompt_template = """\n\nHuman: 다음 텍스트를 간결하게 요약하세오. 텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
        prompt_template = """\n\nHuman: 다음 텍스트를 요약해서 500자 이내로 설명하세오.

        {text}
        
        Assistant:"""        
    else:         
        prompt_template = """\n\nHuman: Write a concise summary of the following:

        {text}
        
        Assistant:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    docs = [
        Document(
            page_content=t
        ) for t in texts[:3]
    ]
    summary = chain.run(docs)
    print('summary: ', summary)

    if summary == '':  # error notification
        summary = 'Fail to summarize the document. Try agan...'
        return summary
    else:
        # return summary[1:len(summary)-1]   
        return summary
              
def get_answer_using_query(query, vectorstore, rag_type):
    wrapper_store = VectorStoreIndexWrapper(vectorstore=vectorstore)        
    
    answer = wrapper_store.query(question=query, llm=llm)    
    print('answer: ', answer)

    return answer

def summerize_text(text):
    docs = [
        Document(
            page_content=text
        )
    ]
    prompt_template = """Write a concise summary of the following:

    {text}
                
    CONCISE SUMMARY """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)
    print('summarized text: ', summary)

    return summary

# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
from langchain.schema import BaseMessage
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer

def get_prompt():
    prompt_template = """\n\nHuman: Using the following <context>, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
    <context>
    {context}
    </context>

    <question>            
    {question}
    </question>

    Assistant:"""

    return PromptTemplate.from_template(prompt_template)

def get_prompt_using_languange_type(query):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)
        
    if word_kor:
        prompt_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
        
        <context>
        {context}
        </context>
            
        <question>            
        {question}
        </question>

        Assistant:"""
    else:
        prompt_template = """\n\nHuman: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
        <context>
        {context}
        </context>
            
        <question>            
        {question}
        </question>

        Assistant:"""
        
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def create_ConversationalRetrievalChain(vectorstore):  
    condense_template = """\n\nHuman: Given the following <history> and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    <history>
    {history}
    </history>
    Follow Up Input: {question}
    
    Assistant: Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    PROMPT = get_prompt()
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 3
            }
        ),         
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        combine_docs_chain_kwargs={'prompt': PROMPT},  

        memory=memory_chain,
        get_chat_history=_get_chat_history,
        verbose=False, # for logging to stdout
        
        #max_tokens_limit=300,
        chain_type='stuff', # 'refine'
        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain                
        return_source_documents=True, # retrieved source (not allowed)
        return_generated_question=False, # generated question
    )
    #qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(qa_prompt_template) 
    #qa.combine_docs_chain.llm_chain.prompt = PROMPT
    #qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template) 
    
    return qa

def extract_chat_history_from_memory(memory_chain):
    chat_history = []
    chats = memory_chain.load_memory_variables({})    
    for dialogue_turn in chats['chat_history']:
        role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
        chat_history.append(f"{role_prefix[2:]}{dialogue_turn.content}")

    return chat_history

def get_revised_question(query):    
    condense_template = """\n\nHuman: Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    
    Assistant: Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template = condense_template, input_variables = ["chat_history", "question"]
    )
    
    chat_history = extract_chat_history_from_memory(memory_chain)
    #print('chat_history: ', chat_history)
    
    question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    return question_generator_chain.run({"question": query, "chat_history": chat_history})

def get_answer_using_template(query, vectorstore, rag_type):        
    #summarized_query = summerize_text(query)        
    #    if rag_type == 'faiss':
    #        query_embedding = vectorstore.embedding_function(summarized_query)
    #        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    #    elif rag_type == 'opensearch':
    #        relevant_documents = vectorstore.similarity_search(summarized_query)
    
    relevant_documents = vectorstore.similarity_search(query)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')
    
    print('length of relevant_documents: ', len(relevant_documents))

    #prompt_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
    #{context}

    #Question: {question}

    #Assistant:"""
    
    #PROMPT = PromptTemplate(
    #    template=prompt_template, input_variables=["context", "question"]
    #)

    PROMPT = get_prompt_using_languange_type(query)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={
                #"k": 3, 'score_threshold': 0.8
                "k": 3
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})
    print('result: ', result)
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(relevant_documents)>=1 and enableReference=='true':
        reference = get_reference(source_documents)
        #print('reference: ', reference)

        return result['result']+reference
    else:
        return result['result']

def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['name']
        page = doc.metadata['row']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

def load_chatHistory(userId, allowTime, memory_chain):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            #memory_chain.save_context({"input": text}, {"output": msg})            
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg) 

def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def lambda_handler(event, context):
    print(event)
    userId  = event['user_id']
    print('userId: ', userId)
    requestId  = event['request_id']
    print('requestId: ', requestId)
    requestTime  = event['request_time']
    print('requestTime: ', requestTime)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)
    print('rag_type: ', rag_type)

    global modelId, llm, vectorstore, isReady, map, qa, memory_chain
    global enableConversationMode, enableReference, enableRAG  # debug
    
    # memory for conversation
    if userId in map:
        memory_chain = map[userId]
        print('memory_chain exist. reuse it!')
    else: 
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True)
        map[userId] = memory_chain
        print('memory_chain does not exist. create new one!')

        allowTime = getAllowTime()
        load_chatHistory(userId, allowTime, memory_chain)

        #conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chain)              
    
    if rag_type == 'opensearch':
        vectorstore = OpenSearchVectorSearch(
            #index_name = "rag-index-*", # all
            index_name = 'rag-index-'+userId+'-*',
            is_aoss = False,
            ef_search = 1024, # 512(default)
            m=48,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url=opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
        )
    elif rag_type == 'faiss':
        print('isReady = ', isReady)
    
    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)
            
            # debugging
            if text == 'enableReference':
                enableReference = 'true'
                msg  = "Referece is enabled"
            elif text == 'disableReference':
                enableReference = 'false'
                msg  = "Reference is disabled"
            elif text == 'enableConversationMode':
                enableConversationMode = 'true'
                msg  = "Conversation mode is enabled"
            elif text == 'disableConversationMode':
                enableConversationMode = 'false'
                msg  = "Conversation mode is disabled"
            elif text == 'clearMemory':
                memory_chain.clear()
                map[userId] = memory_chain
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            elif text == 'enableRAG':
                enableRAG = 'true'
                msg  = "RAG is enabled"
            elif text == 'disableRAG':
                enableRAG = 'false'
                msg  = "RAG is disabled"                
            else:
                if rag_type == 'faiss' and isReady == False: 
                    msg = llm(HUMAN_PROMPT+text+AI_PROMPT)
                else: # opensearch
                    querySize = len(text)
                    textCount = len(text.split())
                    print(f"query size: {querySize}, workds: {textCount}")

                    if querySize<1800 and enableRAG=='true': # max 1985
                        if enableConversationMode == 'true':                                                              
                            if(conversationMothod == 'ConversationalRetrievalChain'):    
                                if isReady==False:
                                    isReady = True
                                    qa = create_ConversationalRetrievalChain(vectorstore)

                                result = qa({"question": text})
                                print('result: ', result)    

                                msg = result['answer']
                                print('\nquestion: ', result['question'])    
                                print('answer: ', result['answer'])    
                                print('chat_history: ', result['chat_history'])    
                                print('source_documents: ', result['source_documents']) 

                                if len(result['source_documents'])>=1 and enableReference=='true':
                                    msg = msg+get_reference(result['source_documents'])

                                # extract chat history
                                chats = memory_chain.load_memory_variables({})
                                chat_history_all = chats['chat_history']
                                print('chat_history_all: ', chat_history_all)
                                
                            else: 
                                revised_question = get_revised_question(text) # generate new prompt using chat history
                                print('revised_question: ', revised_question)
                                msg = get_answer_using_template(revised_question, vectorstore, rag_type) 

                                chat_history_all = extract_chat_history_from_memory(memory_chain) # debugging
                                print('chat_history_all: ', chat_history_all)

                            memory_chain.chat_memory.add_user_message(text)  # append new diaglog
                            memory_chain.chat_memory.add_ai_message(msg)

                        else:
                            msg = get_answer_using_template(text, vectorstore, rag_type)  # using template   
                    else:
                        msg = llm(HUMAN_PROMPT+text+AI_PROMPT)
            
        elif type == 'document':
            object = body
        
            file_type = object[object.rfind('.')+1:len(object)]
            print('file_type: ', file_type)
            
            # load documents where text, pdf, csv are supported
            if file_type == 'csv':
                docs = load_csv_document(object)

                texts = []
                for doc in docs:
                    texts.append(doc.page_content)
                print('texts: ', texts)

            else:
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                'page':i+1
                            }
                        )
                    )        
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))
                                
            if rag_type == 'faiss':
                if isReady == False:   
                    vectorstore = FAISS.from_documents( # create vectorstore from a document
                        docs,  # documents
                        bedrock_embeddings  # embeddings
                    )
                    isReady = True
                else:
                    vectorstore.add_documents(docs)
                    print('vector store size: ', len(vectorstore.docstore._dict))

            elif rag_type == 'opensearch':    
                new_vectorstore = OpenSearchVectorSearch(
                    index_name="rag-index-"+userId+'-'+requestId,
                    is_aoss = False,
                    #engine="faiss",  # default: nmslib
                    embedding_function = bedrock_embeddings,
                    opensearch_url = opensearch_url,
                    http_auth=(opensearch_account, opensearch_passwd),
                )
                new_vectorstore.add_documents(docs)     

                #vectorstore = OpenSearchVectorSearch.from_documents(
                #    docs, 
                #    bedrock_embeddings, 
                #    opensearch_url=opensearch_url,
                #    http_auth=(opensearch_account, opensearch_passwd),
                #)

            # summerize the document
            msg = get_summary(texts)
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)

        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body}, 
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except: 
            raise Exception ("Not able to write into dynamodb")
        
        print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }
