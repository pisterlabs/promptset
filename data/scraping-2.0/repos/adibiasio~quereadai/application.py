from flask import Flask, render_template, redirect, url_for
from authlib.integrations.flask_client import OAuth
import gradio as gr
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.vectorstores import Chroma
import threading
import boto3
from botocore.exceptions import ClientError
import time
import os

def get_secret(secret_name): 
    region_name = "us-east-1"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e
    return get_secret_value_response['SecretString'][get_secret_value_response['SecretString'].index(":")+2:-2]


app = Flask(__name__)
app.secret_key = get_secret("flask_secret_key")

oauth = OAuth(app)
open_ai_api_key = get_secret("OPENAI_KEY")
os.environ['OPENAI_API_KEY'] = open_ai_api_key

chroma_client = chromadb.HttpClient(host='3.84.129.6', port=8000)
openai_embed_function = embedding_functions.OpenAIEmbeddingFunction(open_ai_api_key)

collection = None
pdfname_collection = None
file_output = None


CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#chatbot { flex-grow: 1; overflow: auto; }
#upload { flex-grow: 1; overflow: auto; }
#col1 { height: 92vh !important; }
#col2 { height: 92vh !important; }
#txt { padding-left: 5px; }
footer { display:none !important }
"""


@app.route('/', methods=['GET'])
def index():
    print('awd')
    return render_template('login.html')
 
 
@app.route('/google/')
def google():     
    oauth.register(
        name='google',
        client_id=get_secret("oauth_client_id"),
        client_secret=get_secret("oauth_client_secret"),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={ 
            'scope': 'openid email profile'
        }
    )
    redirect_uri = url_for('google_auth', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)
 
 
@app.route('/google/auth/')
def google_auth():
    token = oauth.google.authorize_access_token()
    global user
    global user_email
    user = token['userinfo']
    user_email = user["given_name"]
    print("TEST")
    print(user_email)
    threading.Thread(target=initialize_gradio).start()
    collection = chroma_client.get_or_create_collection(name=user_email, embedding_function=openai_embed_function)
    return redirect(url_for('gradio'))


def upload_file(files=None):
    collection = chroma_client.get_or_create_collection(name=user_email, embedding_function=openai_embed_function)
    pdfname_collection = chroma_client.get_or_create_collection(name=user_email+"pdf")
    if files:
        for file in files:
            pdfname_collection.add(ids=file.name, embeddings = [0])
            loader = PyPDFLoader(file.name)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
        collection.add(documents=[text.page_content for text in texts], ids=[file.name + str(i) for i in range(len(texts))])
    file_paths = [filename for filename in pdfname_collection.peek(pdfname_collection.count())['ids']]
    return file_paths


def respond(message, chat_history):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    custom_func = OpenAIEmbeddings()
    vectorchrom = Chroma(client=chroma_client, collection_name=user_email, embedding_function=custom_func)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorchrom.as_retriever(), memory=memory)
    chat_history.append((message, qa.run(message)))
    time.sleep(0.5)
    return "", chat_history


def initialize_gradio():
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            with gr.Column(scale=1, elem_id="col1"):
                gr.Markdown(
                f"""
                # Hello, {user.given_name}. 
                # Welcome to Queread AI
                """, elem_id="txt")
                print(upload_file(None))
                file_output = gr.File(value=upload_file(None), elem_id="upload", interactive=False, file_count="multiple")
                upload_button = gr.UploadButton("Click to Upload a File", file_types=["pdf"], file_count="multiple")
                upload_button.upload(upload_file, upload_button, file_output)
            with gr.Column(scale=4, elem_id="col2"):
                chatbot = gr.Chatbot(elem_id="chatbot")
                msg = gr.Textbox()
                clear = gr.ClearButton([msg, chatbot])
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
    demo.launch(server_port=9000)


@app.route('/gradio')
def gradio():
    return render_template('gradio.html')


application = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True)
