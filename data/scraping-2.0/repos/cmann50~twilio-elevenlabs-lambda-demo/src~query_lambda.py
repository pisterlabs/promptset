import boto3
import sys
import os
import json
import utils
from loguru import logger
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

# X-Ray for performance monitoring
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

xray_recorder.configure(service='TwilioElevenLabsDemoQueryLambda')
patch_all()

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}")
ssm = boto3.client('ssm', region_name='us-east-1')


def get_param(key):
    response = ssm.get_parameter(
        Name=f'/{key}',
        WithDecryption=True
    )
    token = response['Parameter']['Value']
    return token


def set_env_vars_from_ssm():
    open_api_key = get_param('umd-aurora-demo-azure-openai-key')
    pg_pass = get_param('postgrespass2')

    if open_api_key is None:
        raise ValueError("umd-aurora-demo-azure-openai-key not in param store.  see setup readme.")
    if pg_pass is None:
        raise ValueError("postgrespass not in param store.  see setup readme.")

    os.environ['OPENAI_API_KEY'] = open_api_key
    os.environ['PG_PASS'] = pg_pass

    logger.debug(
        f"Found vars in ssm store and set to env vars umd-aurora-demo-azure-openai-key, postgrespass")


set_env_vars_from_ssm()


def get_embeddings():
    logger.debug(f"Get API key from env variable.")
    xray_recorder.begin_subsegment('get_embeddings')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key is None:
        raise EnvironmentError("You must specify the OPENAI_API_KEY as an environment variable in the lambda.")

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://umd-dit-gpt.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

    logger.debug(f"Initialize embeddings")
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        chunk_size=1,
        openai_api_key=os.environ["OPENAI_API_KEY"]  # Use the API key from environment variables
    )
    xray_recorder.end_subsegment()
    return embeddings


def create_retriever(embeddings):
    CONNECTION_STRING = f"postgresql://postgres:{os.environ.get('PG_PASS')}@sfsc-demo.cluster-cf3jfbzwbimg.us-east-1.rds.amazonaws.com:5432/sfscdemo"

    logger.debug(f"Connect to vector DB")

    store = PGVector(
        collection_name="sfscdemo",
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    logger.debug(f"Create retriever")
    retriever = store.as_retriever()
    logger.debug(f"Return retriever")

    return retriever


def create_qa_chain(retriever):
    prompt_template = """
    System: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    You MUST respond as if you're speaking to me in a casual, conversational tone. 
    
    YOU MUST use short, natural sentences.
    
    You MUST include filler words and contractions to make your speech sound more human.

    You MUST answer in only one or two sentences. Keep it concise since you are talking on the phone.   After answering the question, ask if there is anything else you can help with.
    

    Context: 
    {summaries}

    Question: 
    {question}

    Your Answer (only one or two sentences, in a casual, conversational tone, with filler words and contractions):
    """
    xray_recorder.begin_subsegment('build_qa_chat')
    logger.debug(f"Create prompt template")
    print('prompt template: ', prompt_template)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    logger.debug(f"Create retrieval QA chain")
    chain_type_kwargs = {"prompt": prompt}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        AzureChatOpenAI(deployment_name="umd-gpt-35-turbo", temperature=0.8), chain_type="stuff",
        retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    xray_recorder.end_subsegment()
    return chain


def do_query(chain, question):
    logger.debug(f"Do query: {question}")
    xray_recorder.begin_subsegment('do_query_chat_gpt')
    xray_recorder.put_metadata('question', question)
    response = chain({"question": question}, return_only_outputs=True)
    xray_recorder.end_subsegment()
    logger.debug(f"Do query response: {response}")
    return response


def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }


def query_chatgpt(question):
    logger.debug(f"Do query: {question}")
    logger.debug(f"Get embeddings")
    embeddings = get_embeddings()
    logger.debug(f"Create retriever")
    retriever = create_retriever(embeddings)
    logger.debug(f"Create QA chain")
    chain = create_qa_chain(retriever)

    logger.debug(f"Question is {question}")

    response = do_query(chain, question)

    answer = response['answer']
    logger.debug(f"Response answer we send back:{answer}")
    logger.debug(f"Response sources with data from vector database:{response}")
    return answer

    # TODO: Get the response text

