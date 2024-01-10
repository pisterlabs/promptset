import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def get_qa_chain():
    load_dotenv()
    # HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    PROMPT = set_the_prompt_template()

    # setting up the memory to use for the qa chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # setup the vectorstore
    vectorstore = get_vector_store(OPENAI_API_KEY)

    # setup the llm
    chain_type = {"prompt": PROMPT}
    # to use open ai, uncomment it
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    # to use models from huggingface, uncomment this
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs=chain_type
    )

    return qa_chain


def set_the_prompt_template():
    prompt_template = """You are a helpful assistant for our restaurant that answers the queries of the customer. You cannot make reservations through chat.
    {context}
    Question: {question}
    Answer here: """

    # setting up the prompt template to use for the qa chain
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


def get_vector_store(OPENAI_API_KEY):
    # get the embeddings for vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # load the vectorstore
    vectorstore = FAISS.load_local("vectorstore", embeddings)
    return vectorstore
