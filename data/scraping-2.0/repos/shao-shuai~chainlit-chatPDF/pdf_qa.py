# Import necessary modules and define env variables
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from chainlit.input_widget import Select, Switch, Slider
from chainlit.types import AskFileResponse
from utils import hash_password
from dotenv import load_dotenv
from typing import Optional
import json
import ocrmypdf

# Password authentication
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
    with open("./auth.json", "r") as file:
        data = json.load(file)
        data[username]
        if data[username] and data[username] == hash_password(str(password)):
            return cl.AppUser(username=username, role="ADMIN", provider="credentials")

        else:
            return None

# Load environment variables
load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")


# text_splitter and system template
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# system prompto to feed LLM
system_template = """Use the following pieces of context to answer the users question.
Please answer the question using the language of question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def store_uploaded_file(file: AskFileResponse):
    """Store an uploaded file to a specified location"""
    file_path = f"./data/{file.name}"
    open(file_path, "wb").write(file.content)
    return file_path

# RAG settings
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=1,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="Chunk_Size",
                label="Chunk Size",
                initial=200,
                min=50,
                max=1000,
                step=10
            ),
            Slider(
                id="Chunk_Overlap",
                label="Chunk Overlap",
                initial=50,
                min=0,
                max=200,
                step=10
            ),
            
        ]
    ).send()

    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements).send()
    

    

@cl.on_settings_update
async def setup_agent(settings):

    files = None
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    # to-do handle cases when multiple files are uploaded
    file = files[0]

    # store file
    file_path = store_uploaded_file(file)

    # Load pdf, the logic is first store file, then load file with PyMuPDFLoader
    docs = PyMuPDFLoader(file_path).load()

    # If the pdf is not OCRed, send a warning
    if docs[0].page_content == "":
        msg = cl.Message(content=f"Warning: The uploaded PDF has not been OCR'ed. Text extraction may not be accurate.")
        await msg.send()

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Split the text into chunks
    texts = text_splitter.split_documents(docs)

    # Change source to page No
    # RetrievalQAWithSourcesChain searches "source" key to return, so we need to reset source if we want to return page No.
    for i, text in enumerate(texts):
        text.metadata['source'] = "chunk-" + str(i) + "-" + "page-"+str(text.metadata['page'])

    # Create metadatas for each chunk
    pages = [f"page-{text.metadata['page']}" for text in texts]
    metadatas = ["chunk-" + str(i) + "-" + pages[i] for i in range(len(pages))]

    # to-do include memroy here

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        texts, embeddings, ids=metadatas
    )

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    
    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message:str):

    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    print(f"this is source {sources}")
    print(dir(res.keys()))
    source_elements = []
    
    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = metadatas
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text.page_content, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()