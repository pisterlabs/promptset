from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import subprocess
import autopep8
import openai
import re
import json
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


load_dotenv()
app = Flask(__name__)
# cors = CORS(app)
CORS(app, resources={r"/": {"origins": ""}})
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_HEADERS'] = 'Content-Type, Access-Control-Allow-Origin'

openai.api_key = os.getenv("OPENAI_API_KEY")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
def run(code, user_input, language):
    fixed_code = autopep8.fix_code(code)
    try:
        command = ['python', '-c', fixed_code]
        if language == 'java':
            command = ['javac', '-',fixed_code]
        elif language == 'javascript':
            command = ['node', '-e', fixed_code]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        output, error = process.communicate(input=user_input, timeout=10)
        if error:
            SystemMessageVariable = f"You are a helpful assistant that assists in fixing software bugs {language} language. You have to solve the code in the {language} language only"
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=SystemMessageVariable
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Generate the response in below structure manner and there should be no code snippets in the suggestion part \n```Suggestions: suggestion``` \nComplete Fixed and Running code in {language}:```code```\n. Given the source code of a software program, there's a known bug that needs fixing. The challenge is to correct the code to resolve the bug without disrupting the existing working solution. Analyze the organization, structure, and dependencies within the source code. Provide suggestions for updating the code to fix the bug while minimizing the impact on the overall functionality. \n\nPython Code:\n\n{code}\n\n."
                    ),
                ]
            )
            llm = OpenAI(temperature=1,model_name="gpt-3.5-turbo-16k")
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
            )
            corrected_code = llm_chain.predict(code=code,language=language)
            start_index = corrected_code.find("Suggestions:")
            end_index_string="Complete Fixed and Running code in "+language+":"
            end_index = corrected_code.find(end_index_string)
            suggestions_part = corrected_code[start_index:end_index].strip()
            code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
            code_matches = re.findall(code_pattern, corrected_code)
            c=''
            for i, code_match in enumerate(code_matches, start=1):
                lines = code_match.split('\n')
                if lines[0].strip() == 'python' or lines[0].strip() == 'java' or lines[0].strip() == 'javascript':
                    lines = lines[1:]
                new_code = '\n'.join(lines)
                c=new_code
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            )
            output, error = process.communicate(input=user_input, timeout=10)
            return {'output': output, 'error': error,'corrected_code':c,'suggestion':suggestions_part,'completeResponse':corrected_code}
        return {'output': output, 'error': error,'suggestion':"Your code ran successfully without any errors!!"}
    except subprocess.TimeoutExpired:
        return {'error': 'Execution timed out'}
    except Exception as e:
        return {'error': str(e)}

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/compile", methods=['POST'])
@cross_origin()
def compile():
    print("compile api called")
    j_data = request.get_json()
    code = j_data.get('code')
    user_input = j_data.get('user_input')
    language = j_data.get('language')
    return jsonify(run(code, user_input, language))

@app.route("/compileAll", methods=['POST'])
@cross_origin()
def compileAll():
    print("compileAll api called")
    j_data = request.get_json()
    code = j_data.get('code')
    user_input = j_data.get('user_input')
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a helpful assistant that assists in fixing software bugs provided extension of the file language. You have to solve the issues of the code in the same language only"
            ),
            HumanMessagePromptTemplate.from_template(
                "Generate the response in below structure manner and suggestions should be generated file name wise and point wise bugs having suggestion code snippet inside ```Snippet```\nSuggestions: \n. You have given json data in which each object represents a file, containing two attributes, name having value as name of the file and code having value as code containing inside the file. There may be bugs in the codes which needs fixing. The challenge is to correct the code to resolve the bug without disrupting the existing working solution. Analyze the organization, structure, and dependencies within the source code. Provide suggestions for updating the code to fix the bug while minimizing the impact on the overall functionality. \n\nPython Code:\n\n{code}\n\n."
            ),
        ]
    )
    llm = OpenAI(temperature=1,model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    res = llm_chain.predict(code=code)
    return {'suggestion':res}

@app.route("/chat", methods=['POST'])
@cross_origin()
def chat():
    j_data = request.get_json()
    question=j_data.get('question')
    code=j_data.get('code')
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a helpful assistant that assists in fixing software bugs. You just answer the question and don't add any other prompt. If the given prompt contains code which have bugs then, generate the corrected code with code explanation"
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template(
                "{additional_messages}"
            ),
        ]
    )
    llm = OpenAI(temperature=1,model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    res=""
    if(code==None):
        res = llm_chain.predict( additional_messages=question)
    else:
        code['file content']=code.pop('user');
        code['user']=question
        code=json.dumps(code)
        res = llm_chain.predict( additional_messages=code)
    return {'chatOutput':res}

if __name__ == "__main__":
    app.run(debug=True)