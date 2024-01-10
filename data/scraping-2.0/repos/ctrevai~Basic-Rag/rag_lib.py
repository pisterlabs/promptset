from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock

# FAISS only work with python3.11

last_uploaded_file = ""


def get_llm():

    model_kwargs = {  # anthropic
        "max_tokens_to_sample": 512,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = Bedrock(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="anthropic.claude-v2:1",  # set the foundation model
        model_kwargs=model_kwargs,)  # configure the properties for Claude

    return llm


pdf_path = "uploaded_file.pdf"


def save_file(file_bytes):
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    return f"Saved {pdf_path}"


def get_index():

    # create embeddings for the index
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1",
    )  # Titan Embedding by default

    loader = PyPDFLoader(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )

    # create the index
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter,
    )

    index_from_loader = index_creator.from_loaders([loader])

    return index_from_loader


# get response from rag client function
def get_rag_response(index, question):
    llm = get_llm()
    response_text = index.query(question, llm=llm)
    return response_text
