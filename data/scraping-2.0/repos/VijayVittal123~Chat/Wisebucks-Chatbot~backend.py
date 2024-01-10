from flask import Flask, request, jsonify
from flask import render_template
from getpass import getpass
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# Get OpenAI API key
# OPENAI_API_KEY = getpass()
# import os
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPENAI_API_KEY = ""
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    
    template = """Financial Query: {question}
Response: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question)
    
    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run()
