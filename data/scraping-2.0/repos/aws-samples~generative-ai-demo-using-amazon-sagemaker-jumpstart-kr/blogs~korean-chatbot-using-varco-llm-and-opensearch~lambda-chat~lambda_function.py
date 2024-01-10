import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.memory import ConversationBufferMemory 

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
endpoint_name = os.environ.get('endpoint_name')
varco_region = os.environ.get('varco_region')
opensearch_url = os.environ.get('opensearch_url')

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
embedding_region = os.environ.get('embedding_region')
endpoint_embedding = os.environ.get('endpoint_embedding')
enableOpenSearch = os.environ.get('enableOpenSearch')
enableReference = os.environ.get('enableReference')
enableRAG = os.environ.get('enableRAG', 'true')

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "text" : prompt, **model_kwargs
        })
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["result"][0]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "request_output_len": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = varco_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

# memory for retrival docs
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer', human_prefix='User', ai_prefix='Assistant')
# memory for conversation
chat_memory = ConversationBufferMemory(human_prefix='User', ai_prefix='Assistant')

# embedding 
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
    region_name = embedding_region,
    content_handler = content_handler2,
)

#embedded_query = embeddings.embed_query("What was the name mentioned in the conversation?")
#print("embedded_query: ", embedded_query[:5])

print('embedding_region: ', embedding_region)
print('endpoint_embedding: ', endpoint_embedding)

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            page_text = page.extract_text().replace('\x00','')
            raw_text.append(page_text.replace('\x01',''))
        contents = '\n'.join(raw_text)            
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
    elif file_type == 'csv':        
        body = doc.get()['Body'].read().decode('utf-8')
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
            
    return texts

def get_answer_using_query(query, vectorstore):
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
    prompt_template = """다음 텍스트를 간결하게 요약하십시오.
텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
    
    TEXT: {text}
                
    SUMMARY:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)
    print('summarized text: ', summary)

    return summary

def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['title']
        page = doc.metadata['document_attributes']['_excerpt_page_number']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

def get_answer_using_template(query, vectorstore):  
    #relevant_documents = vectorstore.similarity_search(query)

    #print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    #print('----')
    #for i, rel_doc in enumerate(relevant_documents):
    #    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    #    print('---')
    
    #print('length of relevant_documents: ', len(relevant_documents))
    
    prompt_template = """다음은 User와 Assistant의 친근한 대화입니다. 
Assistant은 말이 많고 상황에 맞는 구체적인 세부 정보를 많이 제공합니다. 
Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.

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

    if len(source_documents)>=1 and enableReference == 'true':
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

def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global llm, vectorstore, embeddings
    global enableReference, enableRAG, enableConversationMode

    vectorstore = OpenSearchVectorSearch(
        # index_name = "rag-index-*", // all
        index_name = 'rag-index-'+userId+'-*',
        is_aoss = False,
        #engine="faiss",  # default: nmslib
        embedding_function = embeddings,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )

    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body
        
        querySize = len(text)
        print('query size: ', querySize)

        textCount = len(text.split())
        print(f"query size: {querySize}, workds: {textCount}")

         # debugging
        if text == 'enableReference':
            enableReference = 'true'
            msg  = "Referece is enabled"
        elif text == 'disableReference':
            enableReference = 'false'
            msg  = "Reference is disabled"
        elif text == 'enableConversationMode':
            enableConversationMode = 'true'
            msg  = "conversationMode is enabled"
        elif text == 'disableConversationMode':
            enableConversationMode = 'false'
            msg  = "conversationMode is disabled"
        elif text == 'enableRAG':
            enableRAG = 'true'
            msg  = "RAG is enabled"
        elif text == 'disableRAG':
            enableRAG = 'false'
            msg  = "RAG is disabled"
        else:

            if querySize<1800 and enableRAG=='true': 
                answer = get_answer_using_template(text, vectorstore)
            else:
                answer = llm(text)   
            print('answer: ', answer)

            pos = answer.rfind('### Assistant:\n')+15
            msg = answer[pos:]    
        #print('msg: ', msg)
        chat_memory.save_context({"input": text}, {"output": msg})
            
    elif type == 'document':
        object = body
        
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        # load documents where text, pdf, csv are supported
        texts = load_document(file_type, object) 

        pages = len(texts)
        print('pages: ', pages)

        n = 0
        for i in range(int(pages/5+1)):
            docs = []
            if n >= pages: 
                break

            for j in range(5):
                docs.append(
                    Document(
                        page_content=texts[n],
                        metadata={
                            'name': object,
                            'page': n+1
                        }
                    )
                )   
                n = n+1
                if n >= pages: 
                    break

            print(f'docs[{n-1}]: {docs[0]}')    

            new_vectorstore = OpenSearchVectorSearch(
                index_name="rag-index-"+userId+'-'+requestId,
                is_aoss = False,
                #engine="faiss",  # default: nmslib
                embedding_function = embeddings,
                opensearch_url = opensearch_url,
                http_auth=(opensearch_account, opensearch_passwd),
            )
            new_vectorstore.add_documents(docs)   

        # summerization to show the document        
        docs = [
            Document(
                page_content=t
            ) for t in texts[:3]
        ]
        
        # summerization to show the document
        prompt_template = """다음 텍스트를 간결하게 요약하십시오.
텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
    
        TEXT: {text}
                
        SUMMARY:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
        print('summary: ', summary)

        pos = summary.rfind('### Assistant:\n')+15
        msg = summary[pos:] 
                
    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    print('msg: ', msg)

    item = {
        'user-id': {'S':userId},
        'request-id': {'S':requestId},
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
