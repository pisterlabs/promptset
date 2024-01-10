from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import time
from langchain.memory import ConversationBufferMemory

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def query():
    return "Test"

# Defai Bot
#
# Conversation Chain with template prompt and memory
defai_custom_prompt_template = """
The following is a friendly conversation between a human and you.
Your name is Defai and your role is a chatbot that is created by AI Focal. Your task to help the human that you are chatting.
Be polite and professional.
I know you are a LLM,but please pretend to be an oracle chatbot,that can help the human solve any question.
You will analyze the language the user is using and respond in the same language.
Do not be in a hurry to give me a response,Let's Think Step by Step
If you do not know the answer to a question, you truthfully says you do not know.

Current conversation:
{history}
Human: {input}
AI:
"""

defai_prompt = PromptTemplate(input_variables=["history", "input"], template=defai_custom_prompt_template)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
conversation = ConversationChain(
    llm=llm,
    verbose=False,
    prompt = defai_prompt,
)

@app.route("/api/query_defai", methods=["POST"])
def query_defai():
    question = request.form.get("question", "")
    if question:
        try:
            res = conversation.predict(input=question)
            return jsonify({"response" : res}), 200
        except Exception:
            return jsonify({"error": "An error occurred. Please try again!"}), 500
    return jsonify({"error": "Invalid request. Please provide 'question' in form format."}), 400

# Custom bot
qa = None
all_documents = []
vectordb = None
prompt = ""
memory = None

@app.route("/api/bot_initialize", methods=["POST"])
def initialize_chat_bot():
    global all_documents
    global vectordb
    global qa
    global prompt
    global memory

    #Bot Name
    bot_name = request.form.get("bot_name", "")

    #User Prompt
    prompt = request.form.get("prompt", "")

    #URL
    url = request.form.get("website_url", "")
    try:
        if(url):
            loader = WebBaseLoader(url)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(documents)
            all_documents.extend(documents)
    
    except Exception:
        return jsonify({"error": "Invalid URL"}), 500

    #PDF
    pdf_file = request.files.get('pdf')
    try:
        if pdf_file:
            pdf_buffer = pdf_file.read()
            destination_path = store_pdf_buffer(pdf_buffer)
            loader = PyPDFLoader(destination_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(documents)
            all_documents.extend(documents)

    except Exception:
        return jsonify({"error": "Invalid pdf"}), 500

    namePrompt = "You are an AI assistant named {}".format(bot_name)
    # Complete Prompt
    prompt = namePrompt + """
            Use the following pieces of context to answer the question at the end. If you can't find the answer in the given context, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)    

    # VectorDB
    embeddings = OpenAIEmbeddings()
    ids = [str(i) for i in range(1, len(all_documents) + 1)]
    vectordb = Chroma.from_documents(all_documents, embeddings, ids=ids)

    # memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # load model
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm = model,
        chain_type = 'stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory = memory
    )

    message = f"Chatbot called {bot_name} is created successfully"

    response = jsonify({"data": message}), 200
    return response

# store pdf
def store_pdf_buffer(pdf_buffer):
    try:
        # Create the destination folder path
        destination_folder_path = os.path.join(os.getcwd(), 'pdf_file')

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        # Generate a unique file name (you can customize this as needed)
        pdf_file_name = f'pdf_{int(time.time())}.pdf'

        # Create the full path for the PDF file
        destination_path = os.path.join(destination_folder_path, pdf_file_name)

        print(destination_path)
        # Write the PDF buffer to the file
        with open(destination_path, 'wb') as pdf_file:
            pdf_file.write(pdf_buffer)

        return destination_path
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return None

#query custom bot
@app.route("/api/query_bot", methods=["POST"])
def query_chat_bot():
    query =  request.form.get("question", "")
    if(query):
        try:
            result = qa({"question": query})
            return jsonify({"response" : result["answer"]}), 200
        except Exception:
            return jsonify({"error": "An error occurred. Please try again!"}), 500
    return jsonify({"error": "Invalid request. Please provide 'question' in JSON format."}), 400
    
# Reset everything
UPLOAD_FOLDER = "pdf_file"
@app.route("/api/reset_bot", methods=["POST"])
def reset():
    global chat_history
    global all_documents
    global vectordb
    global qa
    global prompt
    global memory
    chat_history = []
    vectordb = None
    qa = None
    memory = None
    prompt = ""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(".pdf"):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                os.remove(file_path)
        embeddings = OpenAIEmbeddings()
        ids = [str(i) for i in range(1, len(all_documents) + 1)]
        vectordb = Chroma.from_documents(all_documents, embeddings, ids=ids)
        i = -1
        while(vectordb._collection.count() >= 1) :
            vectordb._collection.delete(ids=[ids[i]])
            i -= 1
        all_documents = []
        return jsonify({"message": "Reset Successful"})

    except Exception as error:
        print("Error in reset API: ", error)
        return jsonify({"error": str(error)}), 500

if __name__ == '__main__': 
    # app.run()
   app.run(host="0.0.0.0", port=5002, debug=False)
