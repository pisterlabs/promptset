# Imports
import os
import warnings
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Pinecone imports
import pinecone
from langchain.vectorstores import Pinecone

# OpenAI imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Chain imports
from langchain.chains.router import MultiRetrievalQAChain
from langchain.chains import RetrievalQA


# Agent imports
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool

# Memory imports
from langchain.memory.buffer import ConversationBufferMemory

# Flask imports
from flask import Flask, render_template, request, redirect, url_for, jsonify,flash


from langchain.utilities import GoogleSerperAPIWrapper
#from langchain.utilities import SerpAPIWrapper
from langchain.tools import StructuredTool
#from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your_secret_string'

# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
    environment='us-west4-gcp'      
)
# Used for Pinecone.from_existing_index
index_name = "mojosolo-main"
# Used for retrieving namespaces
pineconeIndex = pinecone.Index('mojosolo-main')

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    temperature = 0.1,
    model_name="gpt-4"
)


# client = OpenAI(
#   organization='org-CKhd80ufA54Tg7VoKFi4rXU3',
# )
# client.models.list()

# Initializes the tool lists
tools=[]
toolDescriptions = []

# Memory (currently a Conversation Buffer Memory, will become Motorhead)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up Agent
agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory = memory) # Add verbose = True to see inner workings

# Custom tool for retrieving
def retrievalTool(namespace, name=None, description=None, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    # If they didn't set name/description, use defaults
    if(name == None or name == "Use Default"):
        name="Retrieve from " + namespace + " database"
    if(description == None or description == "Use Default"):
        description = "Useful for when you need to retrieve information from the " + namespace + " database"

    toolDescriptions.insert(id, {"name": name, "description": description, "namespace": namespace, "type": "Retrieval"})

    retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace).as_retriever(search_kwargs={'k': 10})
    
    newTool = create_retriever_tool(
        retriever,
        name,
        description,
    )
    tools.insert(id, newTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory = memory)

# Custom tool for upserting; uses the helper function "upsertToPinecone"
def upsertTool(namespace, name=None, description=None, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    # If they didn't set name/description, use defaults
    if(name == None or name == "Use Default"):
        name="Save to " + namespace + " database"
    if(description == None or description == "Use Default"):
        description = "Useful for when you need to save information to the " + namespace + " database"

    toolDescriptions.insert(id, {"name": name, "description": description, "namespace": namespace, "type": "Upsert"})

    newTool = Tool.from_function(
        func=lambda mem: upsertToPinecone(mem, namespace),
        name=name,
        description=description
    )

    tools.insert(id, newTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory = memory)

def serptool(namespace, name=None, description=None, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    # If they didn't set name/description, use defaults
    if(name == None or name == "Use Default"):
        name="Save to " + namespace + " database"
    if(description == None or description == "Use Default"):
        description = "Useful for when you need to save information to the " + namespace + " database"

    toolDescriptions.insert(id, {"name": name, "description": description, "type": "SERP-Tool"})

    #search = GoogleSerperAPIWrapper()
    search = GoogleSerperAPIWrapper()
    searchTool = StructuredTool.from_function(
        func=search.run,
        name = name,
        description= description,
    )
    tools.insert(id, searchTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory = memory)



# Helper function for upserting to pinecone
def upsertToPinecone(mem, namespace):
    Pinecone.from_texts(texts=[mem], index_name=index_name, embedding=embeddings, namespace=namespace)
    return "Saved " + mem + " to " + namespace + " database"

# Initializes the base tools the agent should start with
# retrievalTool("cust-projectwe-client-pinecone")
# upsertTool("cust-projectwe-client-pinecone")

# Runs Bob
@app.route('/', methods=['POST'])
def mojoBob():
    try:
        # Retrieve the input value from the JSON payload
        input_val = request.json.get("inp")

        response = agent_executor.run(input_val)
        # Return the response in JSON format
        return jsonify({"response": response})
    except Exception as e:
        # Log the error and return a specific error message
        error_message = f"Error executing MojoBob script: {e}"
        print(error_message)
        return jsonify({'error': error_message}), 500

# Helper function that actually does the deletion of a tool
def deleteToolHelper(index):
    tools.pop(index)
    toolDescriptions.pop(index)

# Loads Homepage
@app.route('/', methods=['GET'])
def viewIndex():
    mem=memory.load_memory_variables({})["chat_history"]
    return render_template('index2.html', memory=mem, size=len(mem))

# Loads Namespace CRUD
@app.route('/namespace-crud', methods=['GET'])
def namespace_view_index():
    namespace_data = sorted(pineconeIndex.describe_index_stats()['namespaces'].items())
    return jsonify(namespace_data)

# # Loads Namespace Create View
@app.route('/namespace-create', methods=['GET'])
def namespaceViewCreate():
    return render_template('namespaces/create.html', namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()))

# # Loads Namespace Edit View
@app.route('/namespace-edit/<name>', methods=['GET'])
def namespaceViewEdit(name):
    return render_template('namespaces/edit.html', namespace=name)

# Loads Tool CRUD
@app.route('/tool-crud', methods=['GET'])
def toolViewIndex():
    # Inside your createTool and editTool routes
    
    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory = memory, verbose = True)
    return render_template('tools/crud.html', tools=toolDescriptions)

@app.route('/tool-crud', methods=['GET'])
def tool_view_index():
    # Initialize agent here if it affects the tools data.
    # Replace the following lines with the actual logic to retrieve and initialize your tools.
    tools = get_tool_descriptions()
    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory=memory, verbose=True)

    if not tools:  # If no tools are available after initialization.
        return jsonify({"message": "No tools available"})
    else:
        # Assuming the initialization has now prepared the tools data for response.
        return jsonify(tools)

def get_tool_descriptions():
    # Placeholder function - replace with actual logic to retrieve tools descriptions.
    return []

# # Loads Tool Create View
@app.route('/tool-create', methods=['GET'])
def toolViewCreate():
    # Inside your createTool and editTool routes
    disable_namespace = (type == "SERP-Tool")
    return render_template('create.html', namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()), disable_namespace=disable_namespace)


# # Loads Tool Edit View
@app.route('/tool-edit/<id>', methods=['GET'])
def toolViewEdit(id):
    disable_namespace = (type == "SERP-Tool")
    tool = toolDescriptions[int(id) - 1]
    return render_template('tools/edit.html', id=id, tool=tool, namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()), disable_namespace=disable_namespace)



# # Creates Namespace
@app.route('/namespace-create', methods=['POST'])
def createNamespace():
    namespace = request.form.get("namespace")
    # upserts the namespace to pinecone
    upsertToPinecone(request.form.get("text"), namespace)

    # If they wish to create tools for the given namespace
    if(request.form.get("tools")):
        # Add an upsert tool
        upsertTool(namespace)
        # Add a retrieval tool
        retrievalTool(namespace)
    return redirect(url_for('namespaceViewIndex'))

# # Edits Namespace (Upsert Data)
@app.route('/namespace-edit', methods=['POST'])
def editNamespace():
    upsertToPinecone(request.form.get("text"), request.form.get("namespace"))
    return redirect(url_for('namespaceViewIndex'))

# Deletes Namespace
@app.route('/namespace-crud', methods=['POST'])
def deleteNamespace():
    namespace = request.form.get("namespace")

    # deletes the namespace on pinecone
    pineconeIndex.delete(delete_all=True, namespace=namespace)

    # deletes associated tools
    for idx, tool in enumerate(toolDescriptions):
        if tool["namespace"] == namespace:
            deleteToolHelper(idx)

    return redirect(url_for('namespaceViewIndex'))


@app.errorhandler(500)
def internal_error(error):
    return "500 error: An internal error occurred.", 500

def compile_and_get_function(code, function_name):
    compiled_code = compile(code, "<string>", "exec")
    function_env = {}
    exec(compiled_code, function_env)
    if function_name not in function_env:
        print(function_env.keys())
        raise KeyError(f"The function '{function_name}' is not defined in the provided code.")
    return function_env[function_name]

# # Creates Tool
@app.route('/tool-create', methods=['POST'])
def createTool():
    name = request.form.get("name").strip()  # Ensure to strip any leading/trailing whitespace
    description = request.form.get("description")
    namespace = request.form.get("namespace", "None")  # Default to "None"
    tool_type = request.form.get("type")

    if tool_type == "Custom Tool":
        function_code = request.form.get("function")
        output_schema = {'type': 'object', 'properties': {}, 'required': []}
        property_names = request.form.getlist('property_name[]')
        property_types = request.form.getlist('property_type[]')
        property_descriptions = request.form.getlist('property_description[]')
        property_required = request.form.getlist('property_required[]')

        # Build the output_schema dictionary
        for i, prop_name in enumerate(property_names):
            if prop_name:
                output_schema['properties'][prop_name] = {'type': property_types[i], 'description': property_descriptions[i]}
                if str(i) in property_required:
                    output_schema['required'].append(prop_name)

        try:
            # Parse the code using AST
            parsed_code = ast.parse(function_code, mode='exec')
            # Check that there's exactly one function definition
            if not (len(parsed_code.body) == 1 and isinstance(parsed_code.body[0], ast.FunctionDef)):
                raise ValueError("Code does not define a function properly.")
            
            # Compile and execute the user-provided function code
            compiled_function = compile(parsed_code, "<string>", "exec")
            function_env = {}
            exec(compiled_function, function_env)
            
            # Ensure the function's name matches the one provided by the user
            if name not in function_env or not isinstance(function_env[name], types.FunctionType):
                raise KeyError(f"The function '{name}' is not defined in the provided code.")
                
            function_obj = function_env[name]

            # Placeholder Call to customTool - implement this function
            customTool(function_code, name, description, output_schema, function_obj)
        except SyntaxError:
            flash('Syntax error in function code.', 'error')
            return jsonify({'error': 'Syntax error in function code.'}), 400
        except (ValueError, KeyError) as e:
            flash(str(e), 'error')
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", 'error')
            return jsonify({'error': 'An unexpected error occurred.'}), 500 # Replace 'error_page' with the actual error page route
    # Handle other tool types here as before
    elif tool_type == "Upsert":
        upsertTool(namespace, name, description)  # Implement this function elsewhere
        pass
    elif tool_type == "Retrieval":
        retrievalTool(namespace, name, description)  # Implement this function elsewhere
        pass
    elif tool_type == "SERP-Tool":
        serptool(namespace, name, description)  # Implement this function elsewhere
        pass
    
    return redirect(url_for('toolViewIndex'))


#Edit Tool
@app.route('/tool-edit', methods=['POST'])
def editTool():
    id = int(request.form.get("id")) - 1
    name = request.form.get("name")
    description = request.form.get("description")
    namespace = request.form.get("namespace", "None")  # Set default to "None"
    tool_type = request.form.get("type")

    # Deletes the old tool
    deleteToolHelper(id)

    # Inserts the new tool into the old location
    if tool_type == "Custom Tool":
        function_code = request.form.get("function")
        customTool(function_code, name, description, id)  # Implement the correct handling of custom tools
    elif tool_type == "Upsert":
        upsertTool(namespace, name, description, id)
    elif tool_type == "Retrieval":
        retrievalTool(namespace, name, description, id)
    elif tool_type == "SERP-Tool":
        serptool(namespace, name, description, id)

    return redirect(url_for('toolViewIndex'))

# # Deletes Tool
@app.route('/tool-crud', methods=['POST'])
def deleteTool():
    # Deletes both the tool description and the tool itself
    deleteToolHelper(int(request.form.get("tool")) - 1)
    return redirect(url_for('toolViewIndex'))

import types
import ast

def customTool(function_code, name, description, output_schema, id=len(tools)+1):
    global agent_executor, tools, llm, memory, toolDescriptions
    
    # if tool_id is None:
        # tool_id = len(tools)  # Assuming IDs start at 0
    
    try:
        id=len(tools)+1
        parsed_code = ast.parse(function_code, mode='exec')
        if not (len(parsed_code.body) == 1 and isinstance(parsed_code.body[0], ast.FunctionDef)):
            raise ValueError("Provided code does not define a function properly.")
        
        # print(function_code)
        compiled_function = compile(parsed_code, "<string>", "exec")
        function_env = {}
        exec(compiled_function, function_env)
        
        if name not in function_env:
            raise ValueError(f"Function named '{name}' is not defined.")
        
        function_obj = function_env[name]
        if not isinstance(function_obj, types.FunctionType):
            raise ValueError("Executed code did not result in a Python function.")

        # Assuming you have defined or imported a `Tool` class
        newTool = StructuredTool.from_function(
            func=function_obj,
            name=name,
            description=description
            # output_schema=output_schema
        )
        print(id)
            
        # Insert the new tool description and the tool itself in respective lists
        toolDescriptions.insert(id, {
            "name": name,
            "description": description,
            "namespace": "None",  
            "type": "Custom Tool",  # Adjust the type accordingly
            "output_schema": output_schema
        })
            
        tools.insert(id, newTool)
            
        # Assuming you have defined or imported the `initialize_agent` function correctly
        agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory=memory, verbose=True)
        
    except SyntaxError:
        raise ValueError("Provided function code is not valid Python code.")
    
    except Exception as ex:
        # Catch all other exceptions and handle them appropriately
        print(f"An error occurred while creating a custom tool: {traceback.print_exc()}")
        raise

if __name__ == '__main__':
    app.run(port = 5001, debug=True)