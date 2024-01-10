import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


with open('api_key.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().split('=')[1].strip()
    HUGGINGFACEHUB_API_TOKEN = f.readline().split('=')[1].strip()
    SERPAPI_API_KEY = f.readline().split('=')[1].strip()
    PINECONE_API_KEY = f.readline().split('=')[1].strip()

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)
    
def process_llm_response(llm_response):    
    rtn = llm_response['result']
    rtn += '\n\nSources:\n'
    for source in llm_response["source_documents"]:
        rtn += '   ' + source.metadata['source'] + '\n'

    return rtn

def asking(query):
    llm_response = qa_chain(query)
    return process_llm_response(llm_response)

# which model is longest overall length?
# make a list of all the models and their lengths

demo = gr.Interface(
    fn = asking,
    inputs = gr.Textbox(lines=20, placeholder="Ask."),  # GUI.
    outputs= gr.Textbox(lines=20),
)

demo.launch(share=True)