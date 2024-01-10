import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, make_response, redirect, jsonify
from web3 import Web3
from llama_index import GPTSimpleVectorIndex, download_loader
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.llms import OpenAI
from dotenv import load_dotenv
from eth_account.messages import encode_defunct
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain

load_dotenv()
history = ChatMessageHistory()

env_vars = [
    'OPENAI_API_KEY',
    'SERPAPI_API_KEY',
    'REDDIT_CLIENT_ID',
    'REDDIT_CLIENT_SECRET',
    'REDDIT_USER_AGENT',
    'REDDIT_USERNAME',
    'REDDIT_PASSWORD',
    'ALCHEMY_API_KEY',
]

os.environ.update({key: os.getenv(key) for key in env_vars})
os.environ['WEB3_PROVIDER'] = f"https://polygon-mumbai.g.alchemy.com/v2/{os.environ['ALCHEMY_API_KEY']}"

# Initialize web3
web3 = Web3(Web3.HTTPProvider(os.environ['WEB3_PROVIDER']))

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Initialize chat
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)

# Removed repeated definition of start_query_func and generate_query_func

def create_tool(name, description, index):
    return Tool(
        name=name,
        func=lambda q: index.query(q),
        description=description
    )

# Prepare UnstructuredReader Tool
#UnstructuredReaderClass = download_loader("UnstructuredReader")
#unstructured_reader = UnstructuredReaderClass()
#documents = unstructured_reader.load_data(file=Path('/home/dan/langchain/langchain_bot/cal.txt'))
#ureader_index = GPTSimpleVectorIndex.from_documents(documents)
#ureader = create_tool("Crypto Asset List", "A list of crypto coins and tokens supported in the latest version of Ledger Live. Check the list when asked if a token or coin is supported in Ledger Live.", ureader_index)

# Prepare Reddit tool
subreddits = ['ledgerwallet']
search_keys = []
post_limit = 5

RedditReader = download_loader('RedditReader')
loader = RedditReader()
documents = loader.load_data(subreddits=subreddits, search_keys=search_keys, post_limit=post_limit)
reddit_index = GPTSimpleVectorIndex.from_documents(documents)
reddit_index_tool = create_tool("Reddit", "This is Ledger's subreddit where Ledger users from around the world come to discuss about Ledger products and get support to solve technical issues. Useful to gauge user sentiment about a feature or find the answer to very niche technical issues.", reddit_index)

# Prepare Zendesk tool
#ZendeskReader = download_loader("ZendeskReader")
#loader = ZendeskReader(zendesk_subdomain="ledger", locale="en-us")
#documents = loader.load_data()
#zendesk_index = GPTSimpleVectorIndex.from_documents(documents)
#zendesk = create_tool("Help Center", "This is the Ledger Help Center. Use this tool to find the solution to most technical and shipping issues affecting Ledger products. If you find a helpful article, include the url link to the article in your response", zendesk_index)

# Prepare toolbox
serpapi_tool = load_tools(["serpapi"])[0]
tools = [serpapi_tool, reddit_index_tool]
tools[0].name = "Google Search"

# Run agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Define Flask app
app = Flask(__name__, static_folder='static')

# Define authentication function
def authenticate(signature):
    w3 = Web3(Web3.HTTPProvider(os.environ['WEB3_PROVIDER']))
    message = "Access to chat bot"
    message_hash = encode_defunct(text=message)
    signed_message = w3.eth.account.recover_message(message_hash, signature=signature)
    balance = int(contract.functions.balanceOf(signed_message).call())
    if balance > 0:
        token = uuid.uuid4().hex
        response = make_response(redirect('/gpt'))
        response.set_cookie("authToken", token, httponly=True, secure=True, samesite="strict")
        return response
    else:
        return "You don't have the required NFT!"

# Define function to check for authToken cookie
def has_auth_token(request):
    authToken = request.cookies.get("authToken")
    return authToken is not None

# Define Flask endpoints
@app.route("/")
def home():
    return render_template("auth.html")

@app.route("/auth")
def auth():
    signature = request.args.get("signature")
    response = authenticate(signature)
    return response

@app.route("/gpt")
def gpt():
    if has_auth_token(request):
        return render_template("index.html")
    else:
        return redirect("/")

@app.route('/api', methods=['POST'])
def react_description():
    print(request.json)
    # Get user input from request
    user_input = request.json.get('user_input')  
    memory.chat_memory.add_user_message(user_input)
    
    # Run the OpenAI agent on the user input
    output = agent.run(user_input)
    memory.chat_memory.add_ai_message(output)
    response = conversation.predict(input=user_input)

    # Return the output as JSON
    return jsonify({'output': response})


# Start the Flask app
if __name__ == '__main__':
    app.run(port=8000, debug=True)

