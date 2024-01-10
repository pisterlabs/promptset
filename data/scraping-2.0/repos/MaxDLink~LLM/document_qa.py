
#pip install pypdf 
#export HNSWLIB_NO_NATIVE=1 
#TODO - left off at 1:19:11 from tutorial 
from langchain.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.chat_models import ChatOpenAI 
import chainlit as cl 
from chainlit.types import AskFileResponse 
import os 
import openai

# Read the API key from the local .txt file and assign it to the environment variable
with open("api_key.txt", "r") as file:
    openai_key = file.read().strip()
    
os.environ["OPENAI_API_KEY"] = openai_key #set environment variable for openai api key

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #splits the text from the PDF into chunks of 1000 characters, with 100 characters of overlap between chunks
embeddings = OpenAIEmbeddings(openai_api_key=openai_key) #ADA 2 model (best performance and cheapest cost) for embedding layer defined here 

welcome_message = """Welcome to teh ChainLit PDF QA demo! To get started: 
1. Upload a PDF file
2. Ask a question about the PDF file
"""

def process_file(file: AskFileResponse): 
    import tempfile 

    if file.type == "text/plain": 
        Loader = TextLoader 
    elif file.type == "application/pdf": 
        Loader = PyPDFLoader
    
    with tempfile.NamedTemporaryFile() as tempfile: 
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents) #applies the text splitter to the pdf document 
        for i, doc in enumerate(docs): 
            doc.metadata["source"] = f"source_{i}" #labels each chunk of text with a source number
        return docs 
    
def get_docsearch(file: AskFileResponse): 
    docs = process_file(file)

    #save data in the user session 
    cl.user_session.set("docs", docs) #ensures docs available to both program and client 

    #create a unique namespace for the file 

    docsearch = Chroma.from_documents( #returns relevant query from embeddings 
        docs, embeddings
    )
    return docsearch 

#user interface interactions 
@cl.on_chat_start
async def start(): 
    #sending an image with the local file path 
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None 
    while files is None: #asks for file. Can change what type of file is accepted. 
        #TODO - experiment with CSV files. 
        files = await cl.AskFileMessage(
            content=welcome_message, 
            accept=["text/plain", "application/pdf"],
            max_size_mb=20, 
            timeout=180, 
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Proccessing '{file.name}' ...") #alerts user that file is being processed 
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync 
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type( #unites prompt and model together with chains, but this definition combines prompts llms and other functions (retrieval function -- vector database (docsearch as retreival fucnction))
        ChatOpenAI(temperature=0, streaming=True), 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(max_tokens_limit=4097), #docsearch is the vector database which is the retriever function 
    )

    # Let the user know that the system is ready 
    msg.content = f"'{file.name}' processed. You can now ask questions."
    await msg.update()

    cl.user_session.set("chain", chain) #saves the chain into user session so that the backend can access the chain. 


    #backend: 
    @cl.on_message #whenever a message happens the program fetches the chain 
    async def main(message): 
        chain = cl.user_session.get("chain") #type: RetrievalQAWithSourcesChain 
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True 
        res = await chain.acall(message, callbacks=[cb])

        answer = res["answer"]
        sources = res["sources"].strip()
        source_elements = []

        #LLM's can make up info so providing citations helps the user get accurate info by allwoing them to check the sources 
        #Get the documents from the user session 
        docs = cl.user_session.get("docs")
        metadatas = [doc.metadata for doc in docs]
        all_sources = [m["source"] for m in metadatas]

        if sources: 
            found_sources = [] 

            #Add the sources to the message 
            for source in sources.split(","): 
                source_name = source.strip().replace(".", "")
                #Get the index of the source 
                try: 
                    index = all_sources.index(source_name)
                except ValueError: 
                    continue
                text = docs[index].page_content 
                found_sources.append(source_name)
                #Create the text element referenced in the message 
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources: 
                answer += f"\nSources: {', '.join(found_sources)}"
            else: 
                answer += "\nNo sources found."
            
        if cb.has_streamed_final_answer: 
            cb.final_stream.elements = source_elements 
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=source_elements).send() #sends the answer to the user
