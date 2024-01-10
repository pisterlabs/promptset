import streamlit as st ##from transformers import pipeline
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
import pages.Profile_Page as Profile_Page
import toml
import asyncio
import logging
import threading
import os
import json
import subprocess
from bs4 import BeautifulSoup
from IPython import get_ipython
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_decorators import StreamingContext, llm_prompt, GlobalSettings
from langchain.document_loaders import DirectoryLoader
from flask import Flask, jsonify
from flask import request
from flask import Flask, stream_with_context, request, Response

#st.set_page_config(page_title="BroAi Interface") 


#This is necessary for decorators streaming
GlobalSettings.define_settings(
logging_level=logging.INFO,
print_prompt=True,
print_prompt_name=True)


app = Flask(__name__)

config = toml.load('config.toml')
profile = {}
st.session_state["profile"] = config["Profile"]
profile = st.session_state["profile"]
 
path_from_toml = config["docs"]
repo_path = os.path.normpath( path_from_toml['path'])
model_path_toml = config["model"]
PATH = os.path.normpath( model_path_toml['path'])
digested_toml = config["digested"]
digested_path = os.path.normpath( digested_toml['path'])

bot_reply_enabled = False
bot_process = None

@st.cache_resource 
def llmini():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path=PATH, 
                    n_gpu_layers=43,
                    n_batch=512,
                    n_ctx=5000,
                    f16_kv=True,#thing in case
                    callback_manager=callback_manager,
                    verbose=True,
                    temperature=0.2)
    return llm

llm = llmini()

def escape_path(path):
    return path.replace("\\", "\\\\")

def save(filename, variable,directory):
    # Ensure the directory exists
    directory.replace("\\", "\\\\")
    os.makedirs(directory, exist_ok=True)

    # Combine directory and filename to get the full path
    file_path = os.path.join(directory, filename)

    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Open the file and append the variable to it
        with open(file_path, 'a') as file:
            file.write('\n' + str(variable))  # Adding a newline before the variable for readability
    else:
        # Open the file and write the variable to it
        with open(file_path, 'w') as file:
            file.write(str(variable))

@st.cache_resource
def docloader():
    readme_loader = DirectoryLoader('./documents', glob="**/*.md")
    txt_loader = DirectoryLoader('./documents', glob="**/*.txt")

    loaders = [readme_loader,txt_loader] # add the created loader here
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print (f'You have {len(documents)} document(s) in your data')
    print (f'There are {len(documents[0].page_content)} characters in your document')


    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    

    db = Chroma.from_documents(documents=documents, embedding=HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    ),persist_directory = digested_path) 

    retriever = db.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 1},
    )
    return retriever

doc_db = docloader() # inicia esto luego desde un boton q reciba la ruta y agrega la funcion de descargar

def write_to_config_model(user_input):
    config = {'model': user_input}
    with open('config.toml', 'w') as file:
        toml.dump(config, file)

def write_to_config_docs(user_input):
    config = {'docs': user_input}
    with open('config.toml', 'w') as file:
        toml.dump(config, file)

def write_to_config_digested(user_input):
    config = {'digested': user_input}
    with open('config.toml', 'w') as file:
        toml.dump(config, file)

# Test Api call
@app.route('/llm', methods=['POST'])
def api_endpoint():
    data = request.get_json()
    jobdesc = data.get('jobdesc')
    preferences = data.get("preferences")
    return jsonify({'message': 'Hello from Flask!'})

def run_flask():
    app.run(port=5000)

tokens=[]
def capture_stream_func(new_token:str):
    tokens.append(new_token)

def workPromtInitialization():
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #system prompt
    system_prompt = """Assist a Software Developer to evaluate the job data to identify and recommend remote jobs opportunities that align with 
    the user's qualifications, and preferences. Answer with a definitive decision if the provided job data is suitable for the user. Consider a work unsuitable if it requires IOS or Swift. 
    
    This is the data :
    """

    system_prompt = """ Use the jobdata to decide if it align with the user data and the user expectations. 
    
    This is the data :
    """
    
    instruction = """
    JobData : {context}
    User: {userdata}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context","userdata"],
        template=header_template,
    )
    return QA_CHAIN_PROMPT

def generate_assistant_response(user_input, llm,doc_db):
    #Question about the documents
    docs = doc_db.get_relevant_documents(user_input) 
    chain = load_qa_chain(llm, chain_type="stuff",prompt = llmPrompInitialization() )
    with st.chat_message("assistant"):
        with st.spinner("Working on request"):
            response = chain({"input_documents": docs, "question": user_input},return_only_outputs = True)
            message = {"role": "assistant", "content": response["output_text"]}
            st.write(response["output_text"]) #this is the actual text box on the browser
            st.session_state.messages.append(message["content"]) # after response we set the state again as it was to prevent infinite loop
            return response

async def generate_assistant_work(user_input, llm):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = """ You are a helpful assistant. Assist a Software Developer to evaluate the job data to identify and recommend remote jobs opportunities that align with 
    the user's qualifications, and preferences. You Use the Potential Job data to decide if it align with the User Data. Answer with a definitive decision if the provided job data is suitable for the user. Consider a work unsuitable if it requires IOS or Swift. Answer in Spanish.
    This is the data :
    """  
    instruction = """
    User Data : {context}
    Potential Job: {userdata}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    prompt = ChatPromptTemplate.from_template(header_template)
    chain = prompt | llm    
    response = chain.invoke({"userdata": user_input,"context":profile})
    output = {"output_text":response}
    return output

async def generate_telegram_answer(user_input, llm):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = """ You are a helpful assistant, answer the best possible to the Message
    This is the data :
    """  
    instruction = """
    Message : {message}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    prompt = ChatPromptTemplate.from_template(header_template)
    chain = prompt | llm    
    response = chain.invoke({"message": user_input})
    output = {"output_text":response}
    return output


@app.route('/stream', methods=['GET'])
def stream():
    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def async_function():
            with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
                result = await rizz_assistant(text="and a quote",preferences="say hello")
                print("Stream finished ... we can distinguish tokens thanks to alternating colors")
                return result

        result = loop.run_until_complete(async_function())
        print("---------->"+str(result))

        # 

        yield "Data: " + str(result) + "\n\n"  # Format this as needed for your stream

    return Response(stream_with_context(generate()), content_type='text/event-stream')

#Non functional, Code example to use streams with openai. Use it to combine it wth a local model or custom usecase 
@app.route('/askstream', methods=['POST'])
def streamPostlang():
    def generate():
        data = request.get_json()
        textdesc = data.get('text')
        preferences = data.get("system")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if(preferences == "work"):
           async def async_function():
                with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
                    result = []
                    result = await rizz_assistant(text=textdesc,profile=str(profile))
                    filename = "buffer.txt"
                    add_text_to_file(file_path=filename,text=result)
                    return result

        else:
            async def async_function():
                with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
                    result = []
                    result = await rizz_assistant(text=textdesc,preferences=preferences)
                    return result


        result = loop.run_until_complete(async_function())
        yield "Data: " + str(result) + "\n\n"  # Format this as needed for your stream

    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.route('/ask', methods=['POST'])
def streamPost():
    def generate():
        data = request.get_json()
        textdesc = data.get('text')
        typeofrequest = data.get("system")#change this to request class
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if(typeofrequest == "work"):
           async def async_function():
                result =  await generate_assistant_work(user_input=textdesc,llm=llm)
                return result

        else:
            async def async_function():
                #call another
                return result
            
        result = loop.run_until_complete(async_function())       
        yield result['output_text'] + "\n\n"  # Format this as needed for your stream

    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.route('/telebotchat', methods=['POST'])
def streamPostTelegram():
    def generate():
        data = request.get_json()
        textdesc = data.get('text')
        typeofrequest = data.get("system")#change this to request class
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def async_function():
            result =  await generate_telegram_answer(user_input=textdesc,llm=llm)
            return result
      
        result = loop.run_until_complete(async_function())       
        yield result['output_text'] + "\n\n"  # Format this as needed for your stream

    return Response(stream_with_context(generate()), content_type='text/event-stream')

# Code Example. Function pass through openai, could be used to elevated the problem to a bigger model
@llm_prompt(capture_stream=True) 
async def write_me_short_post(topic:str, platform:str="twitter", audience:str = "developers"):
    """
    Write me a short header for my post about {topic} for {platform} platform.
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass

# Fun prompt. Function pass through openai, could be used to elevate the prompt to a bigger model
@llm_prompt(capture_stream=True) 
async def rizz_assistant( text:str,profile:str):
    """
    You are a Night host, you will deliver witty line openers to make your guest confortable, and a way to achieve this is to banter and use emotion to attract the
    female gender, the objective is to keep her interested in the user. Use the provided messages in the chat to give a sentence to write

This is the user data :

    {profile}

The data from the job is :

    {text} 
    """
    pass
    
async def run_prompt():
    return await write_me_short_post(topic="Hehe, empty prompt") # that can do real magic!

# Code example. It is used when building a Code Interpreter
def exec_python(cell):
    print("USING EXEC")
    ipython = get_ipython()
    result = ipython.run_cell(cell)
    log = str(result.result)
    if result.error_before_exec is not None:
        log += f"\n{result.error_before_exec}"
    if result.error_in_exec is not None:
        log += f"\n{result.error_in_exec}"
    return log

# Code example. Nice to use with Chrome extension
def scrape(url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request

    response = request.post(
        "https://chrome.browserless.io/content?token=YOUR SERPAPI TOKEN", headers=headers, data=data_json) 

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("Content:", text)
        if len(text) < 8000:
            #output = summary(text)
            return text
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def add_text_to_file(file_path, text):
    print("running add_text_to_file")
    try:
        # Try to open the file in append mode
        with open(file_path, 'a') as file:
            file.write(text + '\n')
            print(f'Text added to {file_path}')
    except FileNotFoundError:
        # If the file doesn't exist, create a new file and add text
        with open(file_path, 'w') as file:
            file.write(text + '\n')
            print(f'New file created: {file_path}')


def sidebar():
    """Configure the sidebar and user's preferences."""
    
    with st.sidebar.expander("🔧 Behaviour", expanded=True):
        st.select_slider("Act like a",options=["Assistant","Agent","Expert (need docs)"])

    st.sidebar.divider()

    with st.sidebar:
        with stylable_container(
            key="side",
            css_styles="""div["data-testid="stSidebarUserContent"]{background-color: "#02ab21"}
        """
        ):
            st.container()

        ":green[RESTART APP ON CHANGE]"

        "Model file route"
        user_input = st.text_input("Enter the path:")
        if st.button("Save",key="savebtn"):
            write_to_config_model(user_input)
            st.success("Path saved to config.toml")
            "The model you are using is :"
        st.text(config['model']['path'])

        "Documents file route"
        doc_input = st.text_input("Enter docs path:")
        option = st.radio(
        'Select a document context for:',
        ('Coder','Work Analizer','Character Simulation','Passport Bro AI','Cardinal System',))
        st.write(f'You selected: :green[{option}]')
        options = ['Python','Dart','Kotlin','Javascript']
        country_options = ['Dominican Republic','Philipines','Colombia','Brazil']
        if(option == 'Coder'):
            selected_option = st.selectbox("Choose language for Coder:", options)
            st.write(f"You selected: :green[{selected_option}]")
        elif(option == 'Passport Bro AI'):
            selected_option_country = st.selectbox("Choose Country for Passport Bro AI:", country_options)
            st.write(f"You selected: :blue[{selected_option_country}]")


        if st.button("Load",key="keybtn"):
            write_to_config_docs(doc_input)
            "Context loaded from :"
        st.text( config['docs']['path'])

        "Persistent context saved in folder"
        persistent_input = st.text_input("Digested docs path:")
        if st.button("Digest",key="digestbtn"):
            write_to_config_digested(persistent_input)
            "Digested content saved in :"
        st.text( config['digested']['path'])

        if st.button("Switch page",key="switchbtn"):
            switch_page("Profile Page")


def layout(llm,doc_db):
    global bot_process

    st.title("Telegram Bot Controller")

    # Button to start the bot
    if st.button("Start Bot", key="start_button"):
        if bot_process is None or bot_process.poll() is not None:
            # Start the bot script in a new process
            bot_process = subprocess.Popen(["python", "./chatbots/telegrambotchat.py"])
            st.success("Bot started successfully!")
        else:
            st.warning("Bot is already running!")

    # Button to stop the bot
    if st.button("Stop Bot", key="stop_button"):
        if bot_process and bot_process.poll() is None:
            # Terminate the bot process
            bot_process.terminate()
            st.success("Bot stopped successfully!")
        else:
            st.warning("Bot is not running!")
    
    st.header("Personal :blue[AI]:sunglasses:")
    # System Message

    if st.button('Connect with Bro Ai Extension'):
        thread = threading.Thread(target=run_flask)
        thread.start()
        st.write('Extension connection started!')

    if "messages" not in st.session_state:    #session state dict is the way to navigate the state graphs that you are building
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question"}
        ]
    #is reloading the state on layout
    # User input
    user_input = st.chat_input("Your question") # "Your Question" is only a placeholder and not actually a text input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
    
     # Generate response
    try:
        if st.session_state.messages[-1]["role"] != "assistant": # when the state is not assistant, because there is input, use the model
            generate_assistant_response(user_input,llm,doc_db)     
    except Exception as ex:
        print(str(""))
   
    
def llmPrompInitialization():
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #system prompt
    system_prompt = """You are a helpful coder assistant, you will use the provided context to answer questions.
    Read the given code examples before answering questions and think step by step. If you can not answer a user question based on
    the provided context, inform the user. Do not use any other information for answer to the user"""

    instruction = """
    Context : {context}
    User: {question}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context","question"],
        template=header_template,
    )
    return QA_CHAIN_PROMPT

def mistralPrompInitialization(): 
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #system prompt
    system_prompt = """You are a helpful coder assistant, you will use the provided context to answer questions.
    Read the given code examples before answering questions and think step by step. If you can not answer a user question based on
    the provided context, inform the user. Do not use any other information for answer to the user"""

    instruction = """
    Context : {context} 
    User: {question}"""

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    header_template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context","question"],
        template=header_template, 
    )
    return QA_CHAIN_PROMPT
   
# Toy function, is a challenge to see if you can modify the doc loader to read code
def generate_assistant_coder(user_input, llm,doc_db):
    #Question about the documents
    docs = doc_db.get_relevant_documents(user_input) 
    chain = load_qa_chain(llm, chain_type="stuff",prompt = llmPrompInitialization() )
    with st.chat_message("assistant"):
        with st.spinner("Working on request"):
            response = chain({"context": docs, "question": user_input},return_only_outputs = True)
            message = {"role": "assistant", "content": response["output_text"]}
            st.write(response["output_text"]) #this is the actual text box on the browser
            st.session_state.messages.append(message["content"]) # after response we set the state again as it was to prevent infinite loop
            return response

def telegramlit():
    global bot_reply_enabled

    st.title("Telegram Bot Controller")

    # Button to toggle the bot_reply function state
    if st.button("Toggle Bot", key="toggle_button"):
        bot_reply_enabled = not bot_reply_enabled
        st.write(f"Bot is {'ON' if bot_reply_enabled else 'OFF'}")

def main():
    """Set up user preferences, and layout"""
    sidebar()
    layout(llm,doc_db)

if __name__ == "__main__":
    main()
