from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from flask_cors import CORS  # Import the CORS library
import os
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)

openai_api_key = os.environ.get('OPENAI_KEY')

# Ensure that the OPENAI_KEY is set
if not openai_api_key:
    raise ValueError("The OPENAI_KEY environment variable is not set.")

# Initialize the OpenAI LLM with the API key from the environment variable
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

@app.route('/summarize_essay', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data['text']

    # LangChain summarization logic
        num_tokens = llm.get_num_tokens(text)
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1500, chunk_overlap=50)
        docs = text_splitter.create_documents([text])
        prompt_template = """Write a comprehensive summary of this blog post
        
        {text} 
        
        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT, verbose=True)
        output = chain.run(docs)

        return jsonify({'summary': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)