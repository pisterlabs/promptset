import os
import openai
import pickle
import shutil
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage, download_loader, LLMPredictor
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.readers.database import DatabaseReader


app = Flask(__name__)
CORS(app)

# Get the OpenAI Key from the Env
dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

memory = ConversationBufferMemory(memory_key="chat_history") # Conversation history to make conversation memory possible
llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # Define the Large Language Model as OpenAI and make sure answers are always the same through temperature = 0.


PREFIX = '''You're Wizeline's own Answerbot. Knowledgeable in all regarding the company and how it works. Answering any and all enquiries regarding the company and any associated topic.  
'''

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [WizelineQuestions Repository]
Action Input: the input to the action
Observation: the result of the action
'''

When you have gathered all the information needed to answer the question, just write it to user in a concise yet complete answer. You MUST use the following format to answer to the Human.

'''
Thought: Do I need to use a tool? No
{ai_prefix}: [Write a max 100 word answer that gives a clear answer to the question planted]
'''
"""

SUFFIX = '''

Begin!

Previous conversation history:
{chat_history}

Instructions: {input}
{agent_scratchpad}
'''

keywords = {
    'Founders': ['startup', 'entrepreneurship', 'incubator', 'funding'],
    'Academy': ['education', 'curriculum', 'learning', 'teaching', 'training', 'certification'],
    'Business Operations': ['process', 'improvement', 'efficiency', 'management', 'control'],
    'Engineering': ['CAD', 'development', 'system', 'engineering'],
    'Facilities': ['maintenance', 'building', 'efficiency', 'security', 'planning'],
    'Finance & Accounting': ['financial', 'budgeting', 'forecasting', 'taxation'],
    'Marketing': ['advertising', 'branding', 'content', 'market'],
    'People Operations': ['recruitment', 'employee', 'onboarding', 'talent'],
    'Product': ['product', 'innovation', 'prototyping', 'features'],
    'Sales': ['prospecting', 'networking', 'closing', 'pipeline'],
    'UX Design': ['wireframes', 'prototyping', 'interface', 'aesthetics'],
    'IT & Security Engineering': ['cybersecurity', 'network', 'encryption', 'authentication'],
    'CEO / Exec Staff': ['leadership', 'strategy', 'vision', 'growth'],
    'Delivery': ['supply-chain', 'inventory', 'shipping', 'distribution'],
    'Solutions': ['integration', 'customization', 'automation', 'optimization'],
    'User Experience': ['usability', 'accessibility', 'interaction', 'persona'],
    'Wizeline Questions Staff': ['feedback', 'satisfaction', 'performance', 'communication', 'test'],
    'Legal': ['compliance', 'litigation', 'contracts', 'regulations'],
}


def find_department(query, keywords):
    for department, kws in keywords.items():
        for kw in kws:
            if kw in query:
                return department
    return 'I don\'t know whom to assign it.'

# Create a new global index, or load one from the pre-set path
def initialize_index():
    global stored_docs, index, agent_chain, query_engine
    if os.path.exists('./storage'): # If index exists just load it.
        service_context = ServiceContext.from_defaults(chunk_size_limit=256)
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir='./storage'), service_context=service_context)
        query_engine = index.as_query_engine()
        tool_config = IndexToolConfig(
            query_engine = query_engine,
            name = "WizelineQuestions Repository",
            description = "Useful for answering any question pertaining to Wizeline guidelines, policies, security, etc. If the question pertains to anything involving a Company use this tool to answer.",
            tool_kwargs= {"return_direct": True}
        )
        toolkit = LlamaToolkit(
            index_configs=[tool_config],
        )
        # Generate agent
        agent_chain = create_llama_chat_agent(
            toolkit,
            llm,
            memory=memory,
            verbose=True,
            agent_kwargs={
                'prefix': PREFIX, 
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )
        print("Index Loaded!")
    else: # Create the index from scratch.
        # Query the database for all answers
        query = f"""
        SELECT a.answer_text
        FROM Answers AS a
        """
        DBReader = DatabaseReader(
            scheme = "mysql", # Database Scheme
            host = os.getenv("DB_HOST"), # Database Host
            port = "3306", # Database Port
            user = "admin", # Database User
            password = "wizeq_password", # Database Password
            dbname = os.getenv("DB_NAME"), # Database Name
        )   
        documents = DBReader.load_data(query=query) # Add them to the documents
        DBReader.sql_database.engine.dispose() # Destroys and frees the connection, freeing database resources
        documents += SimpleDirectoryReader('data').load_data() # Load all files in the "data" folder into the documents
        index = GPTVectorStoreIndex.from_documents(documents) # Generate the index
        index.set_index_id("vector_index")
        index.storage_context.persist('./storage') # Store the index for faster loading in future starts of the server

        query_engine = index.as_query_engine()
        # Generate tool to feed Langchain agent
        tool_config = IndexToolConfig(
            query_engine = query_engine,
            name = "WizelineQuestions Repository", # Name of Tool
            # Description, dictates when the tool is used, the context.
            description = "Useful for answering any question pertaining to Wizeline guidelines and policies, and any other thing about the company.",
            #Use to answer any question given as it has been fed Wizeline documents and information and this bot resides in WizelineQuestions, the help forum of Wizeline. Useful if the question pertains to company policy or guidelines regarding the company.",
            tool_kwargs= {"return_direct": True}
        )
        toolkit = LlamaToolkit(
            index_configs=[tool_config],
        )
        # Generate agent
        agent_chain = create_llama_chat_agent(
            toolkit,
            llm,
            memory=memory,
            verbose=True,
            agent_kwargs={
                'prefix': PREFIX, 
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )

# Helper function to upload a file and add it to the documents that feed the bot
@app.route("/api/uploadFile", methods=["POST"])
def upload_file():
    global index, agent_chain
    files = request.files.to_dict()
    try:
        for key, file in files.items():
            filename = secure_filename(file.filename)
            filepath = os.path.join('data', os.path.basename(filename))
            file.save(filepath)
            document = SimpleDirectoryReader(input_files=[filepath]).load_data()[0]
            index.insert(document)
        query_engine = index.as_query_engine()
        # Generate tool to feed Langchain agent
        tool_config = IndexToolConfig(
            query_engine = query_engine,
            name = "WizelineQuestions Repository", # Name of Tool
            # Description, dictates when the tool is used, the context.
            description = "Useful for answering any question pertaining to Wizeline guidelines and policies, and any other thing about the company.",
            #Use to answer any question given as it has been fed Wizeline documents and information and this bot resides in WizelineQuestions, the help forum of Wizeline. Useful if the question pertains to company policy or guidelines regarding the company.",
            tool_kwargs= {"return_direct": True}
        )
        toolkit = LlamaToolkit(
            index_configs=[tool_config],
        )
        # Generate agent
        agent_chain = create_llama_chat_agent(
            toolkit,
            llm,
            memory=memory,
            verbose=True,
            agent_kwargs={
                'prefix': PREFIX, 
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )
        return "Files uploaded!"
    except Exception as e:
        # cleanup temp file
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
        return "Error: {}".format(str(e)), 500

# Helper function to update the index once a question gets answered
@app.route('/api/updateAnswers', methods=['GET'])
def updateAnswers():
    global agent_chain
    # Get latest stored answer in the DB, as it should be the one just created.
    singlequery = f""" 
    SELECT answer_text
    FROM Answers
    ORDER BY createdAt DESC
    LIMIT 1;
    """
    DBReader = DatabaseReader(
        scheme = "mysql", # Database Scheme
        host = os.getenv("DB_HOST"), # Database Host
        port = "3306", # Database Port
        user = "admin", # Database User
        password = "wizeq_password", # Database Password
        dbname = os.getenv("DB_NAME"), # Database Name
    )
    DBAnswer = DBReader.load_data(query=singlequery)[0] # Query the database and get the new question
    DBReader.sql_database.engine.dispose() # Destroys and frees the connection, freeing database resources
    index.insert(DBAnswer)
    query_engine = index.as_query_engine()
    # Generate tool to feed Langchain agent
    tool_config = IndexToolConfig(
        query_engine = query_engine,
        name = "WizelineQuestions Repository", # Name of Tool
        # Description, dictates when the tool is used, the context.
        description = "Always use it to answer any question pertaining to Wizeline guidelines and policies, and any other thing about the company.",
        #Use to answer any question given as it has been fed Wizeline documents and information and this bot resides in WizelineQuestions, the help forum of Wizeline. Useful if the question pertains to company policy or guidelines regarding the company.",
        tool_kwargs= {"return_direct": True}    
    )
    toolkit = LlamaToolkit(
        index_configs=[tool_config],
    )
    # Generate agent
    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True,
        agent_kwargs={
                'prefix': PREFIX, 
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
    )
    return "Answer inserted into Bot Knowledge Base!"
    
# Helper function to update the index once new information gets added
@app.route('/api/updateIndex', methods=['GET'])
def updateIndex():
    if os.path.exists('./storage'):
        shutil.rmtree('./storage')
        initialize_index()
        return "Updated the Index"
    else:
        initialize_index()
        return "Created Index"

# Helper function to delete documents from the repository that the bots feeds itself from.    
@app.route('/api/deleteDoc/<filename>', methods=['DELETE'])
def deleteDoc(filename):
    filepath = os.path.join('data', os.path.basename(filename))
    os.remove(filepath)
    if os.path.exists(filepath):
        return "File does not exist"
    else:
        return "Deleted succesfully"	

# Main function to answer queries, gets a JSON of the whole conversation, deconstructs it to build the answer and return it.
@app.route('/api/pdf_conversation_gpt', methods=['POST'])
def submit_conversation():
    global index
    conversation = request.json
    userInput = conversation[-1]["content"]
    department = find_department(userInput, keywords)
    response = agent_chain.run(userInput)
    if response == "Agent stopped due to iteration limit or time limit.":
        response = "Sorry, your questions wasn't properly processed, could you send it again?"
    answerStruct = {}
    answerStruct["content"] = response
    answerStruct["role"] = "assistant"
    conversation.append(answerStruct)
    return jsonify({'conversation': conversation, 'department': department})
  
CORS(app)
if __name__ == '__main__':
    initialize_index()
    app.run(host='0.0.0.0',port=4000, debug=True)