import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import AmazonKendraRetriever
from langchain.llms.bedrock import Bedrock
import boto3
import toml


PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
PINECONE_ENV = st.secrets.PINECONE_ENV
openai_api_key = st.secrets.OPENAI_API_KEY
kendra_index = st.secrets.KENDRA_INDEX
bedrock_region = st.secrets.AWS_BEDROCK_REGION
kendra_region = st.secrets.AWS_KENDRA_REGION
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
max_tokens = 1024  # Adjust as needed
temperature = 0.7  # Adjust as needed
index_pinecone_hsdemocracy  = 'unidosus-edai-hsdemocracy'
index_pinecone_asu  = 'unidosus-edai-asu'


# Setup bedrock
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_kendra_doc_retriever():
    
    kendra_client = boto3.client("kendra", kendra_region)
    retriever = AmazonKendraRetriever(index_id=kendra_index, top_k=3, client=kendra_client, attribute_filter={
        'EqualsTo': {
            'Key': '_language_code',
            'Value': {'StringValue': 'en'}
        }
    }) 
    return retriever

def embedding_db(index_name):
    # we use the openAI embedding model
    embeddings = BedrockEmbeddings(client=bedrock_client, region_name="us-east-1")
    index_name = 'unidosus-edai-hsdemocracy'
    text_field = "text"
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embeddings, text_field)
    return vectorstore
   
# Function to retrieve answers
def retrieval_answer(query, llm_model, vector_store):        
    # Select the model based on user choice
    if llm_model == 'Anthropic Claude V2':
        model_id = "anthropic.claude-v2"
        model_kwargs = {"max_tokens_to_sample": max_tokens, "temperature": temperature}
        llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)
    elif llm_model == 'Amazon Titan Text Express v1':
        model_id = "amazon.titan-text-express-v1"
        model_kwargs = {"maxTokenCount": max_tokens, "temperature": temperature}
        llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)
    elif llm_model == 'Ai21 Labs Jurassic-2 Ultra':
        model_id = "ai21.j2-ultra-v1"
        model_kwargs = {"maxTokens": max_tokens, "temperature": temperature}
        llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)
    elif llm_model == 'GPT-4-1106-preview':
        llm = ChatOpenAI(model_name="gpt-4-1106-preview",openai_api_key = openai_api_key)

    else:
        return "Invalid LLM model selection."
    
     # Select the Retriever based on user choice
    if vector_store == 'Pinecone: Highschool democracy':
        retriever = embedding_db(index_pinecone_hsdemocracy)
        source = 'Pinecone'
    elif vector_store == 'Pinecone: University of Arizona':
        retriever = embedding_db(index_pinecone_asu)
        source = 'Pinecone'
    elif vector_store == 'Kendra: Highschool democracy':
        retriever = get_kendra_doc_retriever()  
        source = 'Kendra'
    else:
        return "Invalid Vector DB selection."
    #llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)

    if source == 'Pinecone':
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever.as_retriever())
        response = qa(query)
    elif source == 'Kendra':
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa(query)
    return response['result']

