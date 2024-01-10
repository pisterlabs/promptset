import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Dict, List
from io import StringIO
from random import randint
import boto3
import pandas as pd
import json
import os
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain import PromptTemplate
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import SagemakerEndpoint


client = boto3.client('runtime.sagemaker')
aws_region = boto3.Session().region_name
source = []
st.set_page_config(page_title="Document Analysis", page_icon=":robot:")


llm_endpoint_name = os.getenv("nlp_ep_name", default="falcon-7b-instruct-2xl")
embedding_endpoint_name = os.getenv("embed_ep_name", default="huggingface-textembedding-all-MiniLM-L6-v2-2xlarge")



################# Prepare for RAG solution #######################
class SMEmbeddingContentHandler(EmbeddingsContentHandler):
        content_type = "application/x-text"
        accepts = "application/json"        

        def transform_input(self, prompts: List[str], model_kwargs: Dict) -> bytes:
            return prompts[0].encode('utf-8')

        def transform_output(self, output: bytes) -> List[List[float]]:
            query_response = output.read().decode("utf-8")
            
            if isinstance(query_response, dict):
                model_predictions = query_response
            else:
                model_predictions = json.loads(query_response)
    
            translation_text = model_predictions["embedding"]
            return translation_text

class LangchainSagemakerEndpointEmbeddings(SagemakerEndpointEmbeddings):
    def __init__(self, endpoint_name, region_name, content_handler):
        super().__init__(endpoint_name=endpoint_name,
                         region_name=region_name,
                         content_handler=content_handler)

    def embed_documents(self, texts: List[str], chunk_size: int = 1
    ) -> List[List[float]]:
        return super().embed_documents(texts, chunk_size)


region_name = boto3.Session().region_name

embeddings = LangchainSagemakerEndpointEmbeddings(
                endpoint_name=embedding_endpoint_name,
                region_name=region_name,
                content_handler=SMEmbeddingContentHandler())


vector_db_host = os.environ.get("opensearch_vector_db_host", "localhost")
index_name = os.environ.get("opensearch_vector_db_index_name", "default")
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)

docsearch = OpenSearchVectorSearch(
    opensearch_url=vector_db_host,
    embedding_function=embeddings,
    http_auth=auth,
    timeout = 100,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection,
    index_name=index_name,
    engine="faiss",
    bulk_size=1000
)

################# Prepare for chatbot with memory #######################
from langchain.llms.sagemaker_endpoint import LLMContentHandler

class SMLLMContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"text": prompt, "properties" : model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            response = response_json['outputs'][0]["generated_text"].strip()
            if response.rfind('[/INST]') != -1:
                cleaned_response = response[response.rfind('[/INST]')+len('[/INST]'):]
            else:
                cleaned_response = response
            return cleaned_response

model_params = { 
        
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.01,
            "top_k": 100,
            "max_new_tokens": 512,
            "repetition_penalty": 1.03,
}

llm = SagemakerEndpoint(
    endpoint_name=llm_endpoint_name,
    region_name=region_name,
    content_handler = SMLLMContentHandler(),
    model_kwargs = model_params)

@st.cache_resource
def load_rag_chain(endpoint_name: str=llm_endpoint_name):
    condense_question_prompt_template = """<s>
    [INST] Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    ### Chat History
    {chat_history}

    ### Follow Up Input: {question}

    Standalone question:[/INST] """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt_template)

    prompt_template = """<s>[INST] <<SYS>>
    Given the following context, answer the question as accurately as possible:
    <</SYS>>

    ### Question
    {question}

    ### Context
    {context}[/INST] """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    memory_chain = ConversationBufferMemory(memory_key="chat_history", ai_prefix="AI", human_prefix="Human",
                                            input_key="question", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(),
        memory=memory_chain,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
        chain_type='stuff',  # 'refine',
        # max_tokens_limit=300,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa

@st.cache_resource
def load_chain(endpoint_name: str=llm_endpoint_name):
    conversation = ConversationChain(
        llm=llm, verbose=False, memory=ConversationBufferMemory(ai_prefix="AI", human_prefix="Human", input_key="input")
    )
    prompt_template = """<s>[INST] <<SYS>>
You are a helpful assistant. Your objective is to help the user with their questions as best to your knowledge.
<</SYS>>

### Conversation History
{history}

### Question
{input}

### Context
{context}[/INST] """

    # langchain prompts do not always work with all the models. This prompt is tuned for Claude
    llama2_prompt = PromptTemplate.from_template(prompt_template)
    conversation.prompt = llama2_prompt
    return conversation

chatchain = load_chain()
ragchain = load_rag_chain()


# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
if 'rag' not in st.session_state:
    st.session_state['rag'] = False
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))
if 'max_token' not in st.session_state:
    st.session_state.max_token = 512
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1
if 'seed' not in st.session_state:
    st.session_state.seed = 0
if 'option' not in st.session_state:
    st.session_state.option = "NLP"
    
def clear_button_fn():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    st.widget_key = str(randint(1000, 100000000))
    chatchain.memory.clear()
    st.session_state.option = "NLP"
    st.session_state['file_content'] = None


with st.sidebar:
    # Sidebar - the clear button is will flush the memory of the conversation
    st.sidebar.title("Conversation setup")
    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_button_fn)

    # upload file button
    uploaded_file = st.sidebar.file_uploader("Upload a text file", 
                                             key=st.session_state['widget_key'],
                                            )
    if uploaded_file:
        filename = uploaded_file.name
        st.session_state.rag = False
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['widget_key'] = str(randint(1000, 100000000))
        chatchain.memory.clear()

            
    rag = st.checkbox('Use knowledge base (answer question based on the retrieved relevant information from the video data source)', key="rag")

left_column, _, right_column = st.columns([50, 2, 20])

with left_column:
    st.header("Building a multifunctional chatbot with Amazon SageMaker")
    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()
    
    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Input text:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        # when the submit button is pressed we send the user query to the chatchain object and save the chat history
        if submit_button and user_input:
            
            if rag:                    
                output = ragchain.run({"question": user_input})
            else:
                if 'file_content' in st.session_state:
                    output = chatchain.predict(input=user_input, context=st.session_state['file_content'])
                else:
                    output = chatchain.predict(input=user_input, context=None)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
        elif uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = stringio.read().strip()
            st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
            st.session_state['generated'].append("Yes, I have read the file.")
            st.session_state['file_content'] = content
            
        if source:
            df = pd.DataFrame(source, columns=['knowledge source'])
            st.data_editor(df)
            source = []    

    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                

with right_column:
    max_tokens= st.slider(
        min_value=8,
        max_value=1024,
        step=1,
        label="Number of tokens to generate",
        key="max_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        label="Temperature",
        key="temperature"
    )
    
