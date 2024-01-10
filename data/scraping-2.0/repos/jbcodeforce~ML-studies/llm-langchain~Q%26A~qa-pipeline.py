'''
Demonstrate a pipeline using LangChain
'''

from langchain.llms import Bedrock
from langchain.document_loaders import WebBaseLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import bedrock,print_ww
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

CHROMA_DB_FOLDER="./chroma_db"


def loadSourceDocumentsAsChunks():
    '''
    Load a source document and split it into chunks.
    '''
    logger.info("--- Load document, split ---")
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def createVectorStore(bedrock_embeddings, all_splits):
    '''
    Create a vector store from the source document.
    '''
    logger.info("--- Embeddings and save in vector store ---")
    vectorstore = Chroma.from_documents(documents=all_splits, 
                                        embedding=bedrock_embeddings,
                                        persist_directory=CHROMA_DB_FOLDER)

    return vectorstore

def buildPrompt():
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
    return PromptTemplate.from_template(template)

if __name__ == "__main__":
    logger.info(f"--- Connect to Bedrock ---")
    aws_bedrock_client = bedrock.get_bedrock_client()
    bedrock_embeddings = BedrockEmbeddings(client=aws_bedrock_client)
    claude_llm = Bedrock(
                client=aws_bedrock_client,
                model_id="anthropic.claude-v1"
            )
    if not os.path.isdir(CHROMA_DB_FOLDER):
        chuncks = loadSourceDocumentsAsChunks()
        vectorstore = createVectorStore(bedrock_embeddings, chuncks)
    else:
        logger.info("load from vector store")
        vectorstore = Chroma(persist_directory=CHROMA_DB_FOLDER,embedding_function=bedrock_embeddings)

    logger.info("--- Query using llm ---")
    question = "What are the approaches to Task Decomposition?"

    qa_chain = RetrievalQA.from_chain_type(
        claude_llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": buildPrompt()}
        )
    result=qa_chain({"query": question})
    print_ww(result)