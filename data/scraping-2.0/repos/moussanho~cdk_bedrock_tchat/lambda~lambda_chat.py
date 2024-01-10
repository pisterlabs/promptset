import json
import os
import boto3
import uuid
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings

from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.retrievers import AmazonKendraRetriever



REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')


#############Bedrock##############

aws_access_key_id="ASIAYCOZFTGZI5JQ6VXC"
aws_secret_access_key="A8FTz5cts0Za27qfDoINvZ+a/NdTDFBRvQkAOfwS"
aws_session_token="IQoJb3JpZ2luX2VjENL//////////wEaCWV1LXdlc3QtMSJHMEUCIHt5C/qgHyPgo6OuXvi0BTESRGRRKeXt6DMDKNEBMCGWAiEAlZOfDakc5/ivvFMBtnoz69yieFOSm8gLSua1btz8cK4qjAMIu///////////ARABGgw1NTUwNDMxMDExMDYiDATHCnIFEBzdcuj2ICrgAuNYmVqmJXdLhE7+ADWQtOn4lnBTce+Yw3ttM1owVF6UqXtNituZTM4V3fl3JceCqSMODx9SgczZeOkpeBK1heA3AbyiFRvzj6jksmYiYgK/H+XZ5pCMcFj1EbZ9piD/Xbtf/Hac+3v7wiGujK1uQOwuUiwcNDOEpfwI90Hn+nc/URcTBT21vcttgLlYSxyzMSm+v9Zlk3GR+IqMB4utj0dj75MDmz1MJqmZuepyfnfVD4DyL/sE5KmOFvtJHNb/a2L8IE8ed22xLtBnlEnurzOKaTtsIz57qvhVKrExXMbDsmcHC1ZP6YcsyJdPR1aCe/Tb2H0ypTVS0N/FK7Ej+I4aflneuQyO1IXVa0VXO4luv4F3CD8Ok8NFgLF8iNiustNLh8GMEg/EViW0rGyknoGc4JysaSux6voh6hFAuDwPtTlnKeLDhCDuB5PH/hL2X80+9OHz0mjTCGRXzzG/BB4wldOlqAY6pgEnSjsgmTOxMMhEoSdy2ugp5W5XinlL6/z1D7eqIJeXypjKl28pfB0XUotnxeoJs0c21iAAmOm7Aa9gx/1Zkm0F5bipTNe5SaVEI6uF3eZgNmqwWPd4Kv2laqRjlpXyRn9VgEJMi+2H1WHHR/n0oFPlWxxCLlPG1ov1wz14+QildVWgoujh93Iku9Rlqd3QFKFXWV7oE99HVQFZMru/thwXuM8fk9XX"

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
            model_kwargs={"max_tokens_to_sample": 1000},
            model_id = "anthropic.claude-v2")



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

    message_history = DynamoDBChatMessageHistory(table_name="MemoryTableChat", session_id=uuid)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, k=3)

    retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID, top_k=3, client=kendra_client, attribute_filter={
    'EqualsTo': {
        'Key': '_language_code',
        'Value': {'StringValue': 'fr'}
    }
    })
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT, verbose=True)


    response = qa.run(query)   

    return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
