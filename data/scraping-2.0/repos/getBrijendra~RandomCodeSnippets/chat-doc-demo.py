from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import OpenAIModerationChain

from langchain.llms import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool

import chromadb
from chromadb.utils import embedding_functions
from googledrive import CustomGoogleDriveLoader

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document

from operator import itemgetter
from langchain.memory import ConversationBufferMemory

from loguru import logger
from langchain.callbacks import FileCallbackHandler
import asyncio
from langchain.callbacks import get_openai_callback


from typing import Tuple, List

import uuid
import os

import langchain
langchain.debug = True

# logfile = "output.log"

# logger.add(logfile, colorize=True, enqueue=True)
# handler = FileCallbackHandler(logfile)


openai_api_key = ""
openai.api_key = openai_api_key

os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['FOLDER_ID'] = 'werrdw23'

# chroma_db_Client = chromadb.HttpClient(host='localhost', port=8000)
chroma_db_Client = chromadb.HttpClient(host='localhost', port=8000)

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002"
            )


#Loads Data From GoogleDrive
def loadDataFromGoogleDrive():
    folder_id = os.environ.get('FOLDER_ID')
    print(f'FOLDER ID:  {folder_id}')
    loader = CustomGoogleDriveLoader(
        folder_id=folder_id,
        token_path= 'token.json',
        skip_on_failure=True,
        # file_types=["document", "pdf"],
        # file_loader_cls=TextLoader,
        file_loader_kwargs={"mode": "elements"}
        # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
    )
    docs = loader.load()
    print(f'Length of the DOCS: {len(docs)}')
    for doc in docs:
        print(doc.metadata)

    return docs

#Splits the documents list into Chunks
def textChunker(chunk_size: int, chunk_overlap: int, documents: list):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    return docs

#Create OpenAI Embeddings and Save It To Chroma
def createEmbedingsAndSaveToChroma(docs: list):
    # Set up OpenAI embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )
    # load Chroma Client
    chroma_db_Clients = chroma_db_Client
    # Use 'openai_ef' *OpenAIEmbeddings Function* to create the Collection
    collection = chroma_db_Clients.get_or_create_collection(name="my_collection", embedding_function=openai_ef)

    # Save each chunk with the metadata to ChromaDB
    for doc in docs:
        # Save Each Document in chromaDb
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )


def load_data_from_source_to_vstore():
    # load the document and split it into chunks
    loader = TextLoader("./sample_text.txt")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)


    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )

    # load it into Chroma
    persistent_client = chroma_db_Client
    #collection = persistent_client.get_or_create_collection("collection_name")
    collection = persistent_client.get_or_create_collection(name="my_collection", embedding_function=openai_ef)


    for doc in docs:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )

    db = Chroma(
        client=persistent_client,
        collection_name="my_collection",
        embedding_function=embeddings,
    )

    # query it
    query = "How AI is helpful?"
    docs = db.similarity_search(query)

    #print results
    print('length of matching docs:' + str(len(docs)))
    print(docs[0].page_content)




def load_data_from_disk():
    # load from disk
    #persistent_client_for_loading = chromadb.PersistentClient()
    persistent_client_for_loading = chroma_db_Client
    openai_ef_for_loading = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )
    collection = persistent_client_for_loading.get_collection(name="my_collection", embedding_function=openai_ef_for_loading) # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    db2 = Chroma(
        client=persistent_client_for_loading,
        collection_name="my_collection",
        embedding_function=embeddings,
    )
    query2 = "How AI is helpful in climate change?"
    docs2 = db2.similarity_search(query2)

    #print results
    print('################### After loading from disk ##################')
    print('length of matching docs:'+ str(len(docs2)))
    print(docs2[0].page_content)
    return db2





############# USING RAG ################################

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(docs, document_prompt = DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    doc_joined =  document_separator.join(doc_strings)
    print('_combine_documents: doc_joined:', doc_joined)
    return doc_joined


def _format_chat_history(chat_history: List[Tuple]) -> str:
    print('_format_chat_history: chat_history:', chat_history)
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    
    print('_format_chat_history: chat_history combined:', buffer)
    return buffer


def get_tokens_info_for_request(cb):
    return {
        "Total Tokens": cb.total_tokens,
        "Prompt Tokens": cb.prompt_tokens,
        "Completion Tokens": cb.completion_tokens,
        "Total Cost (USD)": cb.total_cost
    }


def answer_queries(user_query):
    result = {}
    with get_openai_callback() as cb:
        db2 = load_data_from_disk()

        moderate = OpenAIModerationChain()

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

        ANSWER_PROMPT = ChatPromptTemplate.from_template(prompt_template)

        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    # search_kwargs={"k": 4}
        retriever = db2.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 4}) 
        memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

        # First we add a step to load memory
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )
        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: _format_chat_history(x['chat_history'])
            } | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0, callbacks=[handler]) | StrOutputParser(),
        }

        print('standalone_question:', standalone_question)

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"]
        }
        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question")
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(callbacks=[handler]),
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer #| moderate

        inputs = {"question": user_query}

        print('Invoking final_chain....')
        result = final_chain.invoke(inputs)
        print(result['answer'].content)
        print(result['docs'])

        # Note that the memory does not save automatically
        # This will be improved in the future
        # For now you need to save it yourself
        memory.save_context(inputs, {"answer": result["answer"].content})
        print(memory.load_memory_variables({}))

    tokens_info = get_tokens_info_for_request(cb)
    return  { 
        "response": result['answer'].content,
        "references": [{"content": doc.page_content, "metadata": doc.metadata} for doc in result['docs']],
        "total_tokens": tokens_info
    }


if __name__ == "__main__":
    # load_data_from_source_to_vstore()

    # Load the documents from Google_DRIVE
    # documents = loadDataFromGoogleDrive()

    # # SPLIT THE TEXT into chunks
    # docs = textChunker(600, 100, documents)

    # # Create OpenAI embeddings And Save it To Chroma
    # createEmbedingsAndSaveToChroma(docs)

    res = answer_queries("You are stupid?")
    # res = answer_queries("Who is SamsungM51?")
    # res = answer_queries("What is Shell Scripting?")
    print("\n\n Result:")
    print(res)


langchain.debug = False