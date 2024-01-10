# Run this script in interactive Python environment to see the UI

import os 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import panel as pn
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Lines 16 to 22 are used to create a panel (UI) which will display text boxes and a run button.
# User will use the upload button to upload the PDF and will enter the query/ques 
# Our code will use the query,vector DB (Chroma) and OPEN AI to find the most similar matches and display

pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width')
pn.state.template.param.update(main_max_width="590px",header_background="#F08080",)
file_input = pn.widgets.FileInput(width=200)
openaikey = pn.widgets.PasswordInput(value="", placeholder="Enter your OpenAI API Key here...", width=500) # creating API text box
prompt = pn.widgets.TextEditor(value="", placeholder="What would you like to know ?", height=50,width=2000, toolbar=False) #creating text prompt
run_button = pn.widgets.Button(name="Run!", height=40,width=20,)  # creating run button
widgets = pn.Row(pn.Column(prompt, run_button, margin=1))

# This function retreives answer to the query posted by the user and returns the answer.
def qa(file, query, chain_type, k):
    
    loader = PyPDFLoader(file)  # load document
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # split the documents into chunks
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()   # selecting the type of embeddings
    
    # create the vectorestore to use as the index
    # Chroma is an open source vector database alternative to Pinecone DB
    db = Chroma.from_documents(texts, embeddings)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k}) # expose this index in a retriever interface
    
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    return result

convos = []  # store all panel objects in a list


# Below function gets answers to query from pdf
def qa_result(_):
    os.environ["OPENAI_API_KEY"] = openaikey.value
    
    # save pdf file to a temp file 
    if file_input.value is not None:
        file_input.save("/content/sample_data/tmp.pdf")    # Be sure to change the path
        prompt_text = prompt.value
        
        if prompt_text:
            result = qa(file="/content/sample_data/tmp.pdf", query=prompt_text)    # Be sure to change the path
            convos.extend([pn.Row(pn.Column(result["result"]))])
         
    return pn.Column(*convos, margin=15, width=1000, min_height=400)

qa_interactive = pn.panel(pn.bind(qa_result, run_button),loading_indicator=True,)

output = pn.WidgetBox('*Output will show up here:*', qa_interactive, height=200, width=2000)

# layout
pn.Column(pn.pane.Markdown("""
    ## Question Answering over PDF
    
    1) Upload a PDF
    2) Enter Your OpenAI API key 
    3) Type a question and click "Run".
    """),
    pn.Row(file_input,openaikey),
    widgets,output

).servable()