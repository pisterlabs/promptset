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

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
opensearch_url = os.environ.get('opensearch_url')
rag_type = os.environ.get('rag_type')
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
isReady = False   
endpoint_llm = os.environ.get('endpoint_llm')
endpoint_embedding = os.environ.get('endpoint_embedding')

enableConversationMode = os.environ.get('enableConversationMode', 'enabled')
print('enableConversationMode: ', enableConversationMode)
enableReference = os.environ.get('enableReference', 'false')
enableRAG = os.environ.get('enableRAG', 'true')

methodOfConversation = 'PromptTemplate' # ConversationalRetrievalChain or PromptTemplate
typeOfHistoryTemplate = 'Basic' # Llam2 or Basic

# Prompt Template
HUMAN_PROMPT = "\n\nUser:"
AI_PROMPT = "\n\nAssistant:"

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

Llama2_BASIC_PROMPT = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{question} [/INST]"""

Llama2_HISTORY_PROMPT = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{chat_history}

<s>[INST] {relevant_docs} [/INST]
<s>[INST] {question} [/INST]"""

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "inputs" : 
            [
                [
                    {
                        "role" : "system",
                        "content" : "You are a kind robot."
                    },
                    {
                        "role" : "user", 
                        "content" : prompt
                    }
                ]
            ],
            "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "max_new_tokens": 1024, 
    "top_p": 0.9, 
    "temperature": 0.1
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_llm, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

map = dict()  # Conversation

# embedding
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Dict, List
class ContentHandler2(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

content_handler2 = ContentHandler2()
embeddings = SagemakerEndpointEmbeddings(
    endpoint_name = endpoint_embedding,
    region_name = aws_region,
    content_handler = content_handler2,
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
        #prompt_template = """\n\nUser: 다음 텍스트를 간결하게 요약하세오. 텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
        prompt_template = """\n\nUser: 다음 텍스트를 요약해서 500자 이내로 설명하세오.

        {text}
        
        Assistant:"""        
    else:         
        prompt_template = """\n\nUser: Write a concise summary of the following:

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

def get_answer_using_template_with_history(query, vectorstore, chat_memory):  
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)
    
    if word_kor:
        condense_template = """\n\nUser: 다음은 User와 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
    
        {chat_history}
        
        User: {question}

        Assistant:"""
    else:
        condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.
        
        {chat_history}
        
        User: {question}

        Assistant:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)     
        
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len)
    texts = text_splitter.split_text(chat_history_all) 

    pages = len(texts)
    print('pages: ', pages)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    
    # load related docs
    relevant_documents = vectorstore.similarity_search(query)
    #print('relevant_documents: ', relevant_documents)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        body = rel_doc.page_content[rel_doc.page_content.rfind('Document Excerpt:')+18:len(rel_doc.page_content)]
        # print('body: ', body)
        
        chat_history = f"{chat_history}\nUser: {body}"  # append relevant_documents 
        print(f'## Document {i+1}: {rel_doc.page_content}')
        print('---')

    print('chat_history:\n ', chat_history)

    # make a question using chat history
    if pages >= 1:
        result = llm(CONDENSE_QUESTION_PROMPT.format(question=query, chat_history=chat_history))
    else:
        result = llm(HUMAN_PROMPT+query+AI_PROMPT)
    # print('result: ', result)

    # add refrence
    if len(relevant_documents)>=1 and enableReference=='true':
        reference = get_reference(relevant_documents)
        # print('reference: ', reference)

        return result+reference
    else:
        return result

def get_history(history):
    msg_history = ""

    # first message
    if history.find('User: ')>=0:
        userMsg = history[history.find('User: ')+6:history.find('Assistant: ')]
        #print('userMsg: ', userMsg)
        history = history[history.find('Assistant: ')+11:len(history)]

        if history.find('User: ')>=0:
            assistantMsg = history[0:history.find('User: ')]
            #print('assistantMsg: ', assistantMsg)
            history = history[history.find('User: '):len(history)]
        else:
            assistantMsg = history[0:len(history)]
            #print('assistantMsg: ', assistantMsg)            
        
        msg_history = userMsg + ' [/INST] '
        msg_history = msg_history + assistantMsg + ' </s>'    
        #print('first history: ', msg_history)
    
    while history.find('User: ')>=0:
        userMsg = history[history.find('User: ')+6:history.find('Assistant: ')]
        #print('userMsg: ', userMsg)
        history = history[history.find('Assistant: ')+11:len(history)]

        if history.find('User: ')>=0:
            assistantMsg = history[0:history.find('User: ')]
            #print('assistantMsg: ', assistantMsg)
            history = history[history.find('User: '):len(history)]
        else:
            assistantMsg = history[0:len(history)]
            #print('assistantMsg: ', assistantMsg)            
        
        msg_history = msg_history + '<s>[INST] ' + userMsg + ' [/INST] '
        msg_history = msg_history + assistantMsg + ' </s>'    
    
    #print('full history: ', msg_history)
    return  msg_history

def get_answer_using_chat_history_and_Llama2_template(query, vectorstore, chat_memory):  
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(Llama2_HISTORY_PROMPT)
        
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len)
    texts = text_splitter.split_text(chat_history_all) 

    pages = len(texts)
    print('pages: ', pages)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    print('chat_history:\n ', chat_history)

    if chat_history.find('User: ') >= 0:
        chat_history = chat_history[chat_history.find('User: '):len(chat_history)]

        history = get_history(chat_history)     
        print('history: ', history)     

    # load related docs
    relevant_documents = vectorstore.similarity_search(query)
    #print('relevant_documents: ', relevant_documents)

    relevant_txt = ""
    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        body = rel_doc.page_content[rel_doc.page_content.rfind('Document Excerpt:')+18:len(rel_doc.page_content)]
        # print('body: ', body)
        
        relevant_txt = relevant_txt + body +'\n'  # append relevant_documents 
        print(f'## Document {i+1}: {rel_doc.page_content}')
        print('---')

    # make a question using chat history
    if pages >= 1:
        result = llm(CONDENSE_QUESTION_PROMPT.format(
            question=query, 
            system_prompt=system_prompt,
            chat_history=history, 
            relevant_docs=relevant_txt))
    else:
        result = llm(query)        
    #print('result: ', result)

    return result    

# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
from langchain.schema import BaseMessage
_ROLE_MAP = {"human": "\n\nUser: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nUser: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer

memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def create_ConversationalRetrievalChain(vectorstore):  
    #condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    #{chat_history}
    
    #User: {question}

    #Assistant:"""
    condense_template = """To create condense_question, given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    # combine any retrieved documents.
    #qa_prompt_template = """\n\nUser: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    #{context}

    #Question: {question}
    
    #Assistant:"""    
    
    qa_prompt_template = """\n\nUser:
    Here is the context, inside <context></context> XML tags.    
    Based on the context as below, answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    <context>
    {context}
    </context>

    User: Use at maximum 5 sentences to answer the following question.
    {question}

    If the answer is not in the context, say "I don't know"

    Assistant:"""  
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),         
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        #combine_docs_chain_kwargs={'prompt': qa_prompt_template},  

        memory=memory_chain,
        get_chat_history=_get_chat_history,
        verbose=False, # for logging to stdout
        
        #max_tokens_limit=300,
        chain_type='stuff', # 'refine'
        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain                
        # return_source_documents=True, # retrieved source (not allowed)
        return_generated_question=False, # generated question
    )
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(qa_prompt_template) 
    
    return qa

def get_answer_using_query(query, vectorstore, rag_type):
    wrapper_store = VectorStoreIndexWrapper(vectorstore=vectorstore)
    
    if rag_type == 'faiss':
        query_embedding = vectorstore.embedding_function(query)
        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    elif rag_type == 'opensearch':
        relevant_documents = vectorstore.similarity_search(query)
    
    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
        print('---')
    
    answer = wrapper_store.query(question=query, llm=llm)
    print(answer)

    return answer

def get_answer_using_template(query, vectorstore, rag_type):        
    #summarized_query = summerize_text(query)        
    #    if rag_type == 'faiss':
    #        query_embedding = vectorstore.embedding_function(summarized_query)
    #        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    #    elif rag_type == 'opensearch':
    #        relevant_documents = vectorstore.similarity_search(summarized_query)
    
    if rag_type == 'faiss':
        query_embedding = vectorstore.embedding_function(query)
        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    elif rag_type == 'opensearch':
        relevant_documents = vectorstore.similarity_search(query)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
        print('---')
    
    print('length of relevant_documents: ', len(relevant_documents))

    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)
    
    if word_kor:
        prompt_template = """\n\nUser: 다음은 User과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
    
        {context}
        
        Question: {question}

        Assistant:"""
    else:
        prompt_template = """\n\nUser: Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.
        
        {context}

        Question: {question}

        Assistant:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
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
        page = doc.metadata['page']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

def load_chatHistory(userId, allowTime, chat_memory):
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

            chat_memory.save_context({"input": text}, {"output": msg})             

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

    global llm, vectorstore, isReady, map, qa
    global enableConversationMode, enableReference, enableRAG  # debug
    
    # memory for conversation
    if userId in map:
        chat_memory = map[userId]
        print('chat_memory exist. reuse it!')
    else: 
        chat_memory = ConversationBufferMemory(human_prefix='User', ai_prefix='Assistant')
        map[userId] = chat_memory
        print('chat_memory does not exist. create new one!')
    
    if rag_type == 'opensearch':
        vectorstore = OpenSearchVectorSearch(
            # index_name = "rag-index-*", // all
            index_name = 'rag-index-'+userId+'-*',
            is_aoss = False,
            embedding_function = embeddings,
            opensearch_url=opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
        )
    elif rag_type == 'faiss':
        print('isReady = ', isReady)
   
    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body

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
        elif text == 'enableRAG':
            enableRAG = 'true'
            msg  = "RAG is enabled"
        elif text == 'disableRAG':
            enableRAG = 'false'
            msg  = "RAG is disabled"
        else:

            if rag_type == 'faiss' and isReady == False: 
                msg = llm(text)
            else: 
                querySize = len(text)
                textCount = len(text.split())
                print(f"query size: {querySize}, workds: {textCount}")
                
                if querySize<1800 and enableRAG=='true': # max 1985
                    if enableConversationMode == 'true':
                        if methodOfConversation == 'PromptTemplate':                            
                            if typeOfHistoryTemplate == "Llama2":
                                msg = get_answer_using_chat_history_and_Llama2_template(text, vectorstore, chat_memory)
                            else:
                                msg = get_answer_using_template_with_history(text, vectorstore, chat_memory)
                                                              
                            storedMsg = str(msg).replace("\n"," ") 
                            chat_memory.save_context({"input": text}, {"output": storedMsg})   

                            allowTime = getAllowTime()
                            load_chatHistory(userId, allowTime, chat_memory)               
                        else: # ConversationalRetrievalChain
                            if isReady==False:
                                isReady = True
                                qa = create_ConversationalRetrievalChain(vectorstore)

                            result = qa(text)
                            print('result: ', result)    
                            msg = result['answer']

                            # extract chat history
                            chats = memory_chain.load_memory_variables({})
                            chat_history_all = chats['chat_history']
                            print('chat_history_all: ', chat_history_all)
                            
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
                    embeddings  # embeddings
                )
                isReady = True                    
            else:                             
                vectorstore.add_documents(docs)
                print('vector store size: ', len(vectorstore.docstore._dict))

        elif rag_type == 'opensearch':         
            new_vectorstore = OpenSearchVectorSearch(
                index_name="rag-index-"+userId+'-'+requestId,
                is_aoss = False,
                embedding_function = embeddings,
                opensearch_url = opensearch_url,
                http_auth=(opensearch_account, opensearch_passwd),
            )
            new_vectorstore.add_documents(docs)    
        
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
