import json
import os
import boto3
import botocore
import uuid
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
# from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings

from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.retrievers import AmazonKendraRetriever

from langchain.prompts import PromptTemplate

REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')


#############Bedrock##############

session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()

aws_access_key_id = credentials.access_key
aws_secret_access_key = credentials.secret_key
aws_session_token= credentials.token

base_session = boto3.Session(
    aws_access_key_id= aws_access_key_id, #your access key
    aws_secret_access_key= aws_secret_access_key,#your secret acces key 
    aws_session_token= aws_session_token
)
base_sts = base_session.client('sts')
bedrock_credentials = base_sts.assume_role(RoleArn = 'arn:aws:iam::743456971407:role/BedrockDataFR', RoleSessionName = str(uuid.uuid4()))['Credentials']
bedrock_session = boto3.Session(
    aws_access_key_id = bedrock_credentials['AccessKeyId'],
    aws_secret_access_key = bedrock_credentials['SecretAccessKey'],
    aws_session_token = bedrock_credentials['SessionToken']
)

bedrock_client = bedrock_session.client('bedrock', region_name = REGION)
##################################
kendra_client = boto3.client("kendra", REGION, 
                                 aws_access_key_id= aws_access_key_id, #your access key
                                 aws_secret_access_key= aws_secret_access_key,#your secret acces key 
                                 aws_session_token= aws_session_token)

llm = Bedrock(client = bedrock_client, 
            model_kwargs={"max_tokens_to_sample":1000,"temperature":1,"top_k":250,"top_p":0.999,"anthropic_version":"bedrock-2023-05-31"},
            model_id = "anthropic.claude-v2")


_template =   """Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know. Always answer in French. Provide very detailed answers and more detail on the documents concerned.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} 
  Answer "don't know" if not present in the document. 

  Assistant:
  """


# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
PROMPT = PromptTemplate(
    template=_template, input_variables=["context", "question"]
)


condense_qa_template = """Human: 
Given the following conversation and a follow up question, rephrase the follow up question 
to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: 
"""

standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

def run_chain(chain, prompt: str):
  return chain({"question": prompt})


def lambda_handler(event, context):
    # body = event
    # body = json.dumps(event)
    body = event['body']
    body = json.loads(body)
    query = body['query']
    uuid = body['uuid']

    message_history = DynamoDBChatMessageHistory(table_name="MemoryTableChat", session_id=uuid)
    # memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, k=3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True,ai_prefix="Assistant", output_key='answer',chat_memory=message_history) # if return_source_documents=True

    retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID, top_k=5, client=kendra_client)

    qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever,
            memory=memory,  
            # condense_question_prompt=standalone_question_prompt, 
            return_source_documents=True, 
            combine_docs_chain_kwargs={"prompt":PROMPT},
            verbose=True)
    

    response = run_chain(qa, query)

    return {
            'statusCode': 200,
            'body': json.dumps(response['answer'],ensure_ascii=False).encode('utf-8').decode('utf-8')
        }