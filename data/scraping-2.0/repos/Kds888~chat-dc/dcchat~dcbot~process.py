import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings 
import os 

pinecone.init(api_key=os.environ.get('pinecone_api_key'),environment=os.environ.get('pinecone_enviroment'))     


def process_input(message):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('open_ai_key'))
    docsearch = Pinecone.from_existing_index(index_name='chat-dc',embedding=embeddings) 
    chat= ChatOpenAI(openai_api_key=os.environ.get('open_ai_key'),verbose=True,temperature=0)
    qa = RetrievalQA.from_llm(llm=chat,retriever=docsearch.as_retriever(),return_source_documents=True) 
    gen_rsponse=qa({'query':message}) # we are not applying chat history right now.
    # links=[]
    # emails=[]
    # for i in range(len(gen_rsponse['source_documents'])): # need to import json file
    #     links.append(json.loads(dict(gen_rsponse['source_documents'][i])['metadata']['metadata'].replace("'","\""))['links'])
    #     emails.append(json.loads(dict(gen_rsponse['source_documents'][i])['metadata']['metadata'].replace("'","\""))['Emails'])
    return gen_rsponse['result']  # +'  ' +"Links with you query: -   " +' ' +' '.join(links) + '  '+"People you can email regading your queries: - "+ '  '+ ' '.join(list(set(' '.join(list(set(emails))).split()))) 