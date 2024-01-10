from flask import Flask, render_template, request
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
import json

app = Flask(__name__)
chat_history = []

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment="us-central1-gcp",
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append(('User', user_input))
        output = process_input(user_input)  # Call your function to process the input text
        chat_history.append(('Chatbot', output))
        return render_template('index.html', chat_history=chat_history)
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        pdf_file = request.files['file']
        file_name = pdf_file.filename

        upload_folder = os.path.join(app.root_path, 'knowledge')
        os.makedirs(upload_folder, exist_ok=True)

        # Generate a unique filename
        pdf_file_name = os.path.join(upload_folder, file_name)
        pdf_file.save(pdf_file_name)

        loader = PyPDFLoader(pdf_file_name)
        document = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1300, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        Pinecone.from_documents(texts, embeddings, index_name="my-knowledgebase")

        return render_template('upload.html')
    else:
        return render_template('upload.html')


@app.route('/summary', methods=['GET', 'POST'])
def summary():
    # Path to the knowledge folder
    knowledge_folder = 'knowledge'

    # List to store file names
    files = []

    if request.method == 'POST':
        # Get the selected file and user input from the request

        selected_file = request.form.get('file')
        user_input = request.form.get('user_input')
        file_path = os.path.join('knowledge', selected_file)

        loader = PyPDFLoader(file_path)
        data = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        prompt_template = user_input + "\n\n" + "{text}" + "\n\n"

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=False,
                                     map_prompt=prompt, combine_prompt=prompt)
        overall_summary = chain({"input_documents": texts}, return_only_outputs=True)
        output = overall_summary.get('output_text')

        # Return the output as a JSON response
        return json.dumps({'output': output})
    else:
        for filename in os.listdir(knowledge_folder):
            # Check if the item is a file
            if os.path.isfile(os.path.join(knowledge_folder, filename)):
                # Add the file name to the list
                files.append(filename)
        return render_template('summary.html', files=files)


def process_input(user_input):
    model = 'gpt-3.5-turbo-16k'

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_existing_index('my-knowledgebase', embeddings)

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name=model,temperature=.7),
                                     chain_type="stuff", retriever=docsearch.as_retriever()
                                     )
    answer = qa({"query": user_input}).get('result')

    return f"Answer: {answer}"


if __name__ == '__main__':
    app.run()
