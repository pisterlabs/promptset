import os
from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)




app = Flask(__name__)

# Set API key for OpenAI Service
os.environ['OPENAI_API_KEY'] = 'your_key'

# Create an instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Define the path to the default PDF file on the server
DEFAULT_PDF_PATH = './sample.pdf'

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the JSON data from the request body
        data = request.get_json()

        # Check if 'prompt' is in the JSON data
        if 'prompt' in data:
            prompt = data['prompt']
        else:
            return jsonify({'error': 'Missing "prompt" in request JSON data'}), 400

        # Load the default PDF file
        loader = PyPDFLoader(DEFAULT_PDF_PATH)
        pages = loader.load_and_split()
        store = Chroma.from_documents(pages, embeddings, collection_name='default_report')

        vectorstore_info = VectorStoreInfo(
            name="default_report",
            description="Default annual report as a PDF",
            vectorstore=store
        )

        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )

        # Process the prompt
        response = agent_executor.run(prompt)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
