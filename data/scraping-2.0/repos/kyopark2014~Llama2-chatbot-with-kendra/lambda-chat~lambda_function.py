import boto3
import json
import datetime
import sys
import os
import time
import PyPDF2
import csv
import re
from io import BytesIO

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')
endpoint_name = os.environ.get('endpoint')
kendra_region = os.environ.get('kendra_region')

enableConversationMode = os.environ.get('enableConversationMode', 'enabled')
print('enableConversationMode: ', enableConversationMode)
enableReference = os.environ.get('enableReference', 'false')
enableRAG = os.environ.get('enableRAG', 'true')

conversationMothod = 'PromptTemplate' # ConversationalRetrievalChain or PromptTemplate
isReady = False   

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
    "max_new_tokens": 512, 
    "top_p": 0.9, 
    "temperature": 0.6
} 
HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

map = dict()  # Conversation

kendra = boto3.client("kendra", region_name=aws_region)
retriever = AmazonKendraRetriever(
    index_id=kendraIndex,
    region_name=aws_region,
    client=kendra
)

# store document into Kendra
def store_document(s3_file_name, requestId):
    documentInfo = {
        "S3Path": {
            "Bucket": s3_bucket,
            "Key": s3_prefix+'/'+s3_file_name
        },
        "Title": s3_file_name,
        "Id": requestId
    }

    documents = [
        documentInfo
    ]
        
    result = kendra.batch_put_document(
        Documents = documents,
        IndexId = kendraIndex,
        RoleArn = roleArn
    )
    print(result)

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
              
def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['title']
        page = doc.metadata['document_attributes']['_excerpt_page_number']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

def get_answer_using_template_with_history(query, chat_memory):  
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)
    
    if word_kor:
        condense_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
    
        {chat_history}
        
        Human: {question}

        Assistant:"""
    else:
        condense_template = """\n\nHuman: Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.
        
        {chat_history}
        
        Human: {question}

        If the answer is not in the context say "주어진 내용에서 관련 답변을 찾을 수 없습니다."

        Assistant:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
            
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
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
    relevant_documents = retriever.get_relevant_documents(query)
    #print('relevant_documents: ', relevant_documents)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        body = rel_doc.page_content[rel_doc.page_content.rfind('Document Excerpt:')+18:len(rel_doc.page_content)]
        # print('body: ', body)
        
        chat_history = f"{chat_history}\nHuman: {body}"  # append relevant_documents 
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

memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def create_ConversationalRetrievalChain():  
    #condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    #{chat_history}
    
    #Human: {question}

    #Assistant:"""
    condense_template = """To create condense_question, given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    # combine any retrieved documents.
    #qa_prompt_template = """\n\nHuman: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    #{context}

    #Question: {question}
    
    #Assistant:"""    
    
    qa_prompt_template = """\n\nHuman:
    Here is the context, inside <context></context> XML tags.    
    Based on the context as below, answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    <context>
    {context}
    </context>

    Human: Use at maximum 5 sentences to answer the following question.
    {question}

    If the answer is not in the context say "주어진 내용에서 관련 답변을 찾을 수 없습니다."

    Assistant:"""  
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever,         
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

def get_answer_using_template(query):
    relevant_documents = retriever.get_relevant_documents(query)
    print('length of relevant_documents: ', len(relevant_documents))

    if(len(relevant_documents)==0):
        return llm(HUMAN_PROMPT+query+AI_PROMPT)
    else:
        print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
        print('----')
        for i, rel_doc in enumerate(relevant_documents):
            print(f'## Document {i+1}: {rel_doc.page_content}.......')
            print('---')

        # check korean
        pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
        word_kor = pattern_hangul.search(str(query))
        print('word_kor: ', word_kor)
        
        if word_kor:
            prompt_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
        
            {context}
            
            Human: {question}

            Assistant:"""
        else:
            prompt_template = """\n\nHuman: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            {context}
            
            Human: {question}

            Assistant:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa({"query": query})
        print('result: ', result)

        source_documents = result['source_documents']        
        print('source_documents: ', source_documents)

        if len(source_documents)>=1 and enableReference == 'true':
            reference = get_reference(source_documents)
            # print('reference: ', reference)

            return result['result']+reference
        else:
            return result['result']

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

    global llm, kendra, map, isReady, qa
    global enableConversationMode, enableReference, enableRAG  # debug
    
    # memory for conversation
    if userId in map:
        chat_memory = map[userId]
        print('chat_memory exist. reuse it!')
    else: 
        chat_memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
        map[userId] = chat_memory
        print('chat_memory does not exist. create new one!')

        allowTime = getAllowTime()
        load_chatHistory(userId, allowTime, chat_memory)

    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body
        querySize = len(text)
        print('query size: ', querySize)

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
            
            if querySize<1000 and enableRAG=='true': 
                if enableConversationMode == 'true':
                    if conversationMothod == 'PromptTemplate':
                        msg = get_answer_using_template_with_history(text, chat_memory)
                                                              
                        storedMsg = str(msg).replace("\n"," ") 
                        chat_memory.save_context({"input": text}, {"output": storedMsg})                  
                    else: # ConversationalRetrievalChain
                        if isReady==False:
                            isReady = True
                            qa = create_ConversationalRetrievalChain()

                        result = qa(text)
                        print('result: ', result)    
                        msg = result['answer']

                        # extract chat history
                        chats = memory_chain.load_memory_variables({})
                        chat_history_all = chats['chat_history']
                        print('chat_history_all: ', chat_history_all)
                else:
                    msg = get_answer_using_template(text)
            else:
                msg = llm(HUMAN_PROMPT+text+AI_PROMPT)
            
                
    elif type == 'document':
        object = body
                    
        # store the object into kendra
        store_document(object, requestId)

        # summerization to show the content of the document
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