import os
import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

#OpenAI API Key goes here
os.environ["OPENAI_API_KEY"] = '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'

def get_document():
    loader = TextLoader('context.txt')
    data = loader.load()
    return data

my_data = get_document()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
my_doc = text_splitter.split_documents(my_data)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(my_doc, embeddings)
retriever=vectordb.as_retriever(search_type="similarity")

template = """{question}"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question"])

def generate_response(query,chat_history):
    if query:
        llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
        my_qa = ConversationalRetrievalChain.from_llm(llm, retriever, QA_PROMPT)
        result = my_qa({"question": query, "chat_history": chat_history})

    return result["answer"]

# Create a user interface
def chat(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    output = generate_response(input,history)
    history.append((input, output))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    state = gr.State()
    text = gr.Textbox(placeholder="Question")
    
    text.submit(chat, inputs=[text, state], outputs=[chatbot, state])

demo.launch(share = True)