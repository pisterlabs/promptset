import os
import io
import glob
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain 
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from flask_socketio import SocketIO, emit
from pathlib import Path
from datetime import datetime
# %%
from langchain.chains.question_answering import load_qa_chain
# %%
# Set your API key
os.environ["OPENAI_API_KEY"] = "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"

# %%
data_folder = os.path.join(os.getcwd(), 'data')
file_folder = os.path.join(os.getcwd(), 'uploads')

#  %%
# This is a processing step and assumes that folders and files need to be processes and created. This will use Tokes
embeddings = OpenAIEmbeddings()

# This will load the files and create the documents, this is a processing step
loader = DirectoryLoader(file_folder,loader_cls=UnstructuredFileLoader)
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
persist_directory = os.path.join(data_folder, 'multiple')
db = Chroma.from_documents(docs, embeddings,persist_directory=persist_directory)
db.persist()

# %%
# To load the database use the following, this is a loading step if the processing step has already been done

# This should almost be a function
data_folder = os.path.join(os.getcwd(), 'data')
file_folder = os.path.join(os.getcwd(), 'uploads')
embeddings = OpenAIEmbeddings()
persist_directory = os.path.join(data_folder, 'multiple')

# %%
# This is the Load db code
vectordb= Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever()

# %%
llm = OpenAI()

# %%
from langchain.chains import RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="map_reduce", retriever=retriever)
# %%
chain({"question": "What is the effect of cannaboids on psychiatric symptoms"}, return_only_outputs=True)



# %%
#This is messy code for debuging the prompt length
def count_tokens(string):
    delimiters = [' ', '\n']
    tokens = []
    
    current_token = ""
    for char in string:
        if char in delimiters:
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    token_count = len(tokens)
    return token_count


# %%
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=250)

# #initiate memory so it won't be empty
# memory.save_context({"input":" "},{"output": " "})
# template the Prompt
from langchain import PromptTemplate
from IPython.display import Markdown, display
template = """

I want you to act as a research expert in your field that can understand research papers and answer questions about them. If possible also give instances where there are research gaps.  

Here is the question: "{user_question}"

Here is the History of the conversation: {memory}

Use numbered list and new lines if needed to organize data, if the user asks for a list, or you have more than one concept, in a clear format. Give clear and concise answers with a good balance between formal scientific and conversational structure. If I aks you to cite sources try to do so AMA style.

"""
prompt = PromptTemplate(input_variables= ["user_question","memory"], template=template)

while True:
# Get the Question
    user_question = input("Enter a question: ")
    
    if user_question == "exit":
        break
    prompt_output = prompt.format(user_question=user_question, memory=memory.load_memory_variables({}))

    
    import textwrap
    wrapper = textwrap.TextWrapper(width=50)  
    # Set the maximum width for a line
    answer = chain({"question": prompt_output},   return_only_outputs=False)['answer']
    
    wrapped_page_content = wrapper.fill(text=answer)
    display(print(wrapped_page_content))
    
    memory.save_context({"input":user_question},{"output": answer})
    display(count_tokens(str(memory.load_memory_variables({}))))
        
        
        
        