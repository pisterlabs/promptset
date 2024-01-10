# Imports
from fastbook import *
from fastai.vision.widgets import *
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import gradio as gr

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY_HERE"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "good question!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Process a given URL and set up the QA chain
def setup_chain(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return qa_chain

# Gradio Interface Function
def process_input(url, query):
    qa_chain = setup_chain(url)
    result = qa_chain({"question": query})
    answer = result.get("answer", "No answer found.")
    citations = "\n".join(result.get("source_documents", []))
    return f"Answer: {answer}\n\nCitations:\n{citations}"

# Create the Gradio Interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.inputs.Textbox(placeholder="Enter a URL..."), gr.inputs.Textbox(placeholder="Ask a question...")],
    outputs=gr.outputs.Textbox(label="Answer with Citations"),
    live=True,
    capture_session=True,
)

# Launch the Gradio Interface
iface.launch(share=True)
