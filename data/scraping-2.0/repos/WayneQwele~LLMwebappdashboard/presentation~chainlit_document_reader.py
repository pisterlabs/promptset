
import os

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
import openai
from chainlit.types import AskFileResponse

import chromadb 

from getkey import getkey

getkey()

#export HNSWLIB_NO_NATIVE = 1

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """ 
Welcome to the Chainlit PDF QA demo!

1) Upload a PDF file
2) Ask a question about a file

"""


def process_file(file: AskFileResponse):
    """ 
    Chainlit offers this as boiler plate code
    """
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    
    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

def get_docsearch(file: AskFileResponse):
    """ 
    Retrieve our data from the loaded embeddings
    """
    docs = process_file(file)

    #Save data in the user session
    cl.user_session.set("docs",docs)
    #Create a unique namespace for the file

    docsearch = Chroma.from_documents(
                docs, embeddings
    )
    return docsearch


# Chainlit functions
@cl.on_chat_start
async def start():
    # sending an image to an local file host
    await cl.Message(content= "You can now query your documents/pds").send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content = welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb = 20,
            timeout=180
        ).send()

    files = files[0] # Authors naming is not consistent here!

    msg = cl.Message(content=f"Proccessing '{files.name}'")

    await msg.send()

    # No aysn implementation  in Pineclone client

    docsearch = await cl.make_async(get_docsearch)(files) # this calls the 2nd function defined.

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type ="stuff",
        retriever = docsearch.as_retriever(max_tokens_limit=4097)
    )

    #Let the user know that the system is ready
    msg.content = f"{files.name}: is processed. You may ask questions."
    await msg.update()

    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message):
    """ 
    
    """
    chain = cl.user_session.get("chain") # type: RetrievalQAWithSourcesChain

    answer_prefix_tokens=["FINAL", "ANSWER"]
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=answer_prefix_tokens,
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # get the documents from the user session 

    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources =[]

        # Add sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)

            #Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))
        
        if found_sources:
            answer += f"\nSources: {','.join(found_sources)}"
        else:
            answer += "\nNo of Sources Found :("

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
        

    





