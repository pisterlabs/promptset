import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import pinecone
import numpy as np
from langchain.vectorstores import Pinecone
from pymongo.mongo_client import MongoClient
from pymongo.mongo_client import MongoClient
import itertools

def chunks2(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

uri = st.secrets["mongo"]
# Create a new client and connect to the server
client = MongoClient(uri)
db=client.Images
table=db.Imagebase

load_dotenv()
pinecone.init(api_key=st.secrets["pinecone"], environment="us-west1-gcp-free")

index_name = 'chattypdf'

with st.sidebar:
    st.title("ChattyPDF")
    st.markdown('''
                About:
                An LLM powered chatbot built using:
                - Streamlit
                - LangChain
                - OpenAI''')
    add_vertical_space(5)

    st.write("Made by [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)")

def main():
    
    index = pinecone.Index(index_name)

    st.header("Uploaded files: ")

    index_stats_response = index.describe_index_stats()
    st.write(index_stats_response.namespaces)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
    model=model_name)

    text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

    choice=st.selectbox('What would you like to do?', ['Select from dropdown','Upload PDFs', 'Delete File','Partially delete PDF','Ask Question',"Upload Image"],index=0)

    if choice=='Partially delete PDF':
        name=st.text_input("Enter name of PDF:")
        txt=st.file_uploader("Upload txt file",type="txt")
        st.write("OR")
        txt=st.text_input("Enter text")
        if txt and name:
            parts = text_splitter.split_text(text=txt)
            ember2 = embed.embed_documents(parts)
            sims=index.query(ember2, top_k=1000,namespace=name)
            ids = []
            scores = []
            for i in sims.matches:
                if i["score"]>=0.5:
                    ids.append(i['id'])
                    scores.append(i['score'])
            index.delete(ids=ids,namespace=name)
            st.write([(x,y) for (x,y) in zip(ids,scores)],"deleted")

    elif choice=='Upload Image':
        image=st.file_uploader("Upload your image",type=['png', 'jpeg', 'jpg'])
        title=st.text_input("Upload title of image",max_chars=1000)
        desc=st.text_input("Upload description (optional)",max_chars=1000)
        if image and title:
            if desc:
                emberd = embed.embed_documents(desc)

                index.upsert(vectors=zip([title],emberd),namespace="Images")
            else:
                embert=embed.embed_documents(title)
                index.upsert(vectors=zip([title],embert),namespace="Images")
            template={
                    "id": title,
                    "img": image.getvalue(),
                    }
            table.insert_one(template)
            st.write(title,"uploaded")

    # elif choice=="Image Search":
    #     txt=st.text_input("Enter text")
    #     if txt:
    #         parts = text_splitter.split_text(text=txt)
    #         ember2 = embed.embed_documents(parts)
    #         sims=index.query(ember2, top_k=1,namespace="Images")
    #         pred=sims.matches[0]
    #         title=pred["id"]
    #         que=table.find_one({"id":title})
    #         pic=que['img']
    #         st.image(pic)
    #         st.write(title)


    elif choice=='Upload PDFs':
        uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True,type="pdf")
        if uploaded_files is not None:
            for pdf in uploaded_files:
                pdf_reader=PdfReader(pdf)
                text=""
                for page in pdf_reader.pages:
                    text+=page.extract_text()
                
                chunks = text_splitter.split_text(text=text)

                m=len(chunks)
                store_name=pdf.name[:-4]
                
                ember = embed.embed_documents(chunks)

                if m>=1000:
                    ids = map(str, np.arange(m).tolist())
                    idx=chunks2(ids,batch_size=100)
                    for ids_vectors_chunk,id in zip(chunks2(ember, batch_size=100),idx):
                        index.upsert(vectors=zip(id,ids_vectors_chunk),namespace=store_name)

                else:
                    ids = map(str, np.arange(m).tolist())
                    index.upsert(vectors=zip(ids,ember),namespace=store_name)
                st.write(store_name,"uploaded")


    elif choice=='Ask Question':

        vectorstore = Pinecone(
            index, embed.embed_query,"text"
        )

        llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")
        
        ruff = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                
        tools = [
        Tool(
        name="PDF System",
        func=ruff.run,
        description="useful for when you need to answer questions about user given PDF file. Input should be a fully formed question.",
        ),]

        prefix = """Have a conversation with a human, answering the following questions as elaborately as you can. Answer in steps if possible. You have access to the following tools:"""
        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        message_history = RedisChatMessageHistory(
        url="redis://default:zyRNg3pQk44tfbpw4fQauy1lsuacbDdA@redis-19775.c10.us-east-1-4.ec2.cloud.redislabs.com:19775", ttl=600, session_id="user"
        )

        memory = ConversationBufferWindowMemory(k=5,
            memory_key="chat_history", chat_memory=message_history
        )
        
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        query=st.text_input("Ask question about your files: ")
        if query:
            res=agent_chain.run(input=query)
            st.write(res)
            chunks = text_splitter.split_text(text=query)
            ember2 = embed.embed_documents(chunks)
            sims=index.query(ember2, top_k=1,namespace="Images")
            pred=sims.matches[0]
            title=pred["id"]
            que=table.find_one({"id":title})
            pic=que['img']
            st.image(pic)
            st.write(title)

    elif choice=="Delete File":
        name=st.text_input("Enter namespace")
        if name:
            index.delete(delete_all=True,namespace=name)
            st.write(name,"deleted")
            if name=="Images":
                db.Imagebase.delete_many({})

if __name__=='__main__':
    main()
