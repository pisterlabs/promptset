from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from utils import *

import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO

from dotenv import load_dotenv
import shutil
import subprocess
import psutil
import platform


print("The operating system is:", platform.system())

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

system_template = """Use the following pieces of context to answer the users question.
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
    HumanMessagePromptTemplate.from_template("{question}")
]

choices = ['summary', 'answer', 'sources']


prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def on_chat_start():

    elements = [
        cl.Image(name="image1", display="inline", path="./robot.jpg")
    ]

    await cl.Message(content = "Hello there. Welcome to AskAnythingBot", elements = elements).send()
    files = None
    
    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload the PDF File to begin!",
            accept = ['application/pdf'],
            max_size_mb = 20,
            timeout = 1000
        ).send()

    file = files[0]

    msg = cl.Message(content = f"Processing {file.name}")
    await msg.send()

    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""

    for page_number in range(len(pdf.pages)):
        pdf_text += pdf.pages[page_number].extract_text()
    
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl-0", "documents": file.name} for i in range(len(texts))]
    list_docs = [file.name]

    # Create a Chroma vector store
    # embeddings = HuggingFaceEmbeddings(model_name= 'all-MiniLM-L6-v2')
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas, persist_directory = './database'
    )

    retriever = docsearch.as_retriever(search_type = "similarity",
                            search_kwargs = {"k": 3})   

    llm = ChatOpenAI(temperature = 0.1)
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever= retriever,
    )
    # chain = load_qa_chain(llm,  chain_type="stuff")
    id_sources = {'id': len(texts), 'num_sources': 1}

    # Save the metadata and texts in the user session
    cl.user_session.set("llm", llm)
    cl.user_session.set("list_docs", list_docs)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    cl.user_session.set("docsearch", docsearch)
    cl.user_session.set("metadata_index", id_sources)
    cl.user_session.set("embeddings", embeddings)
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    
    llm = cl.user_session.get("llm")
    chain = cl.user_session.get("chain")
    list_docs = cl.user_session.get("list_docs")
    id_sources = cl.user_session.get("metadata_index")
    embeddings = cl.user_session.get("embeddings")

    print(list_docs)

    

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True


    res = await chain.acall(message.content, callbacks = [cb])
    answer = res["answer"]
    sources = res["sources"].strip()

    # print(answer)
    

    source_elements = []

    # Get the metadata and texts from user sessino
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m['source'] for m in metadatas]
    texts = cl.user_session.get("texts")


    if sources:
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace('.', "")
            source_name = source_name.replace(' ', '')

            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            
            print(source_name)
            print(index)

            text = texts[index]
            found_sources.append(source_name)

            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    final_answer = cl.Message(content = "")
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:   
        # await cl.Message(content=answer, elements=source_elements).send()
        for chunk in answer.split(" "):
            await final_answer.stream_token(chunk + " ")
        
        final_answer.elements = source_elements
        await final_answer.send()


    while True:
        options1 = await cl.AskActionMessage(
            content = "What action do you want the chatbot to perform?",
            actions = [
                cl.Action(name = "summary", value = "summary", display = "inline", text = "Get Summary"),
                cl.Action(name = "query", value = "query", display = "inline", text = "Get Query"),
                cl.Action(name = "upload", value = "upload", display = "inline", text = "Upload Sources"),
            ],
            timeout = 1000
        ).send()

        print(options1.get('value'))
        if options1.get('value')  == 'summary':


            list_actions = []
            for i in range(len(list_docs)):
                list_actions.append(
                    cl.Action(name = list_docs[i], value = f"{i}", display = "inline", text = list_docs[i])
                )
                
            list_actions.append(
                cl.Action(name = "other_file", value = "other_file", 
                                display = "inline",
                                text = "Upload another file")
            )
            
            options2  = await cl.AskActionMessage(
                content = "Choose the file you have uploaded",
                actions = list_actions,
                timeout = 1000
            ).send()

            if options2.get('value') == 'other_file':
                

                texts_local, metadata_local, filename = await summarize(
                                llm,
                                id_sources = id_sources,
                                embeddings = embeddings)
                
                texts.extend(texts_local)
                metadatas.extend(metadata_local)
                list_docs.append(filename)

                id_sources['id'] += len(texts)
                id_sources['num_sources'] += 1

                cl.user_session.set("metadata_index", id_sources)
                cl.user_session.set("texts", texts)
                cl.user_session.set("metadatas", metadatas)
                cl.user_session.set("list_docs", list_docs)

            else:
                docs = list_docs[int(options2.get('value'))]
                await summarize_one_file(llm, docs)


        elif options1.get('value') == 'query':
            break

        elif options1.get('value') == 'upload':
            texts_local, metadata_local, filename = await upload_file(
                                id_sources = id_sources,
                                embeddings = embeddings)
            
            texts.extend(texts_local)
            metadatas.extend(metadata_local)
            list_docs.append(filename)

            id_sources['id'] += len(texts)
            id_sources['num_sources'] += 1

            cl.user_session.set("metadata_index", id_sources)
            cl.user_session.set("texts", texts)
            cl.user_session.set("metadatas", metadatas)
            cl.user_session.set("list_docs", list_docs)
        
        
@cl.on_chat_end
def end():
    print("Good Bye!")

    current_process = psutil.Process(os.getpid())
    print("current process terminated: ", current_process)
    # This will kill the process in Powershell Windows
    # IN Unix operating systems, it sends the signal SIGTERM to the process
    current_process.terminate()
