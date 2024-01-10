
import asyncio
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from .schemas import ChatHistory
from langchain.embeddings import OpenAIEmbeddings
import os
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.chat_models import ChatOpenAI
from text_generation import Client

API_KEY = "a4eee9dd-ba9d-41b3-b0a7-0cd6b069ff76"
start_key="sk-ykOlX7M"
end_key="ZmpNaOJ79EB"
os.environ['OPENAI_API_KEY']=f'{start_key}ZvQqFXZckMN54T3BlbkFJiTJ32jnPh{end_key}'
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=API_KEY,
              environment="gcp-starter")
              
index_name = 'dataindex'

index = pinecone.Index(index_name)

# def generate_prompt(prompt: str, system_prompt: str ) -> str:
#     return f"""
# [INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# {prompt} [/INST]
# """.strip()

async def chatStreamingResponse(chat:ChatHistory,llm,docsearch):
    try:
       
        callback = AsyncIteratorCallbackHandler()
        openai_key = os.getenv('OPENAI_API_KEY')
        # SERVER_URL="https://qfle7l1jguvllr-80.proxy.runpod.net"
        model = ChatOpenAI(openai_api_key=openai_key, callbacks=[callback], temperature=0.6,streaming=True)
        
        response=""
        Chat = [[]]
        # chat_history = "".join([f"User:{data.User} \n\n AI:{data.AI} \n\n" for data in chat.History])
        context = docsearch.similarity_search(chat.UserMessage)
        content ="".join([d.page_content for d in context])
        # client = Client(SERVER_URL, timeout=60)
        # text = ""
        # prompt =generate_prompt(system_prompt=f"You are a helpful assistant your job is to read context and answer to user. ### Context:{content} \n\n. You can also read chat history to get context just for user question answer. ### Chat History:{chat_history}",prompt=chat.UserMessage)
        # for response in client.generate_stream(prompt, max_new_tokens=512):
        #     if not response.token.special:
        #         new_text = response.token.text
        #         print(new_text, end="")
        #         text += new_text
        #         yield new_text
    
        for data in chat.History:
            ls = [
                HumanMessage(content=data.User),
                AIMessage(content=data.AI)]
            Chat[0].extend(ls)
        Chat[0].extend([HumanMessage(content=f"Must Read all this context and then answer to user. ### Context:{content} \n\n ### User Message:{chat.UserMessage}")])

    
        
        
        task = asyncio.create_task(model.agenerate(messages=Chat))
        try:
            async for token in callback.aiter():
                print(token)
                response+= token
                yield token
            

        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            callback.done.set()
        await task
    except Exception as e:
            print(f"Caught exception: {e}")







def add_embeddings(file):
    file = file.document
    if file:
        pdf = file.read()
        pdf_file = BytesIO(pdf)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page in range(num_pages):
            text += pdf_reader.pages[page].extract_text()

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1500,
    chunk_overlap=150)

    record_texts = text_splitter.split_text(text)
    vectors=[]
    
    record_metadatas = [{
        "chunk": j, "text": text, "file_name":file.filename
    } for j, text in enumerate(record_texts)]
    embeds = embeddings.embed_documents(record_texts)
    count=0
    for record_metadata in record_metadatas:
        vectors.append({'id':f"{record_metadata['chunk']}", 'values':embeds[count],'metadata':{'text':record_metadata['text'],'file_name':record_metadata['file_name']}})
        count+=1
    index.upsert(vectors)

    return {"status": "success"}


