import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
# from kendra.kendra_index_retriever import KendraIndexRetriever
from langchain.retrievers import AmazonKendraRetriever


REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')
SM_ENDPOINT_NAME = os.environ.get('SM_ENDPOINT_NAME')

# Generative LLM 


    
# SageMaker langchain integration, to assist invoking SageMaker endpoint.
llm=SagemakerEndpoint(
    endpoint_name=SM_ENDPOINT_NAME,
#    model_kwargs=kwargs,
    region_name=REGION,
    content_handler=content_handler, 
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def lambda_handler(event, context):
    print(event)
    body = json.loads(event['body'])
    print(body)
    query = body['query']
    uuid = body['uuid']
    print(query)
    print(uuid)

    message_history = DynamoDBChatMessageHistory(table_name="MemoryTable", session_id=uuid)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, k=3)
    
    # This retriever is using the query API, self implement
    # retriever = KendraIndexRetriever(kendraindex=KENDRA_INDEX_ID, 
    #                                  awsregion=REGION, 
    #                                  return_source_documents=True)
    
    # This retriever is using the new Kendra retrieve API https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/
    retriever = AmazonKendraRetriever(
        index_id=KENDRA_INDEX_ID,
        region_name=REGION,
    )
    
    # retriever.get_relevant_documents(query)
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

    response = qa.run(query)   
    clean_response = response.replace('\n','').strip()

    return {
        'statusCode': 200,
        'body': json.dumps(f'{clean_response}')
        }
