from flask import Flask
from flask import request, send_file
import json
app = Flask(__name__)

def azure_openai_quest(question):
    #Note: The openai-python library support for Azure OpenAI is in preview.
    import os
    import openai

    try:
        response = openai.ChatCompletion.create(
          engine="qitest1",
          messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":question}],
          temperature=0.7,
          max_tokens=800,
          top_p=0.95,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None)
        return response['choices'][0]['message']['content']
    except:
        return "Request Failed"

    
def azure_doc_search_init(doc):
    # set the environment variables needed for openai package to know to reach out to azure
    import os

    from langchain.embeddings import OpenAIEmbeddings
    # For Azure API, the chunk_size must be 1, for public openai, it's OK
    embeddings = OpenAIEmbeddings(deployment="qitestemb1", chunk_size = 1)

    from langchain.llms import AzureOpenAI
    from typing import List
    class NewAzureOpenAI(AzureOpenAI):
        stop: List[str] = None
        @property
        def _invocation_params(self):
            params = super()._invocation_params
            # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
            params.pop('logprobs', None)
            params.pop('best_of', None)
            params.pop('echo', None)
            #params['stop'] = self.stop
            return params

    llm = NewAzureOpenAI(deployment_name='qitest1', temperature=0.9)


    from langchain.document_loaders.csv_loader import CSVLoader


    loader = CSVLoader(file_path=doc)
    data = loader.load()
    # 10 lines as test
    data = data[:7000]

    from langchain.schema import Document
    from langchain.vectorstores import Chroma
    vectorstore = Chroma.from_documents(data, embeddings)

    retriever = vectorstore.as_retriever()

    from langchain.chains import RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    
    return qa

def azure_conversation_summary_init():
    from langchain.llms import AzureOpenAI
    import os


    from langchain.callbacks.base import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # llm = AzureOpenAI(deployment_name="qitest1")
    from langchain.llms import AzureOpenAI
    from typing import List
    class NewAzureOpenAI(AzureOpenAI):
        stop: List[str] = None
        @property
        def _invocation_params(self):
            params = super()._invocation_params
            # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
            params.pop('logprobs', None)
            params.pop('best_of', None)
            params.pop('echo', None)
            #params['stop'] = self.stop
            return params

    llm = NewAzureOpenAI(deployment_name='qitest1', temperature=0.9)


    from langchain.prompts import PromptTemplate
    from langchain import PromptTemplate, LLMChain

    prompt_template="""Please summarize following text in few sentence. 
    the text:
    {text}
    concise summary in english:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm_chain = LLMChain(prompt=PROMPT, llm=llm)
    return llm_chain

summary_chain = azure_conversation_summary_init()
doc_qa = azure_doc_search_init('./ref_bus.csv')


@app.route("/")
def hello_world():
    answer = llm_chain.run("Hello World")
    return f"<p>{answer}</p>"

# curl -X POST  http://10.128.0.171:9888/post_question -d question='who is Jeo Biden'
@app.route('/post_question', methods=['POST'])
def any_question():
    data = request.form
    question = data['question']
    r = llm_chain.run(question)
    return r

# curl -X POST  http://10.128.0.27:9888/post_openai_question -d question='who is Jeo Biden'
@app.route('/post_openai_question', methods=['POST'])
def any_openai_question():
    data = request.form
    question = data['question']
    r = azure_openai_quest(question)
    return r

# This is a demo API for now, curl -X POST  http://10.128.0.27:9888/post_openai_doc_question -d question='what are the Data Warehouse Name that has amount information'
@app.route('/post_openai_doc_question', methods=['POST'])
def post_openai_doc_question():
    data = request.form
    question = data['question']
    response = doc_qa({"query": question}, return_only_outputs=True)
    answer = response['result']
    r = answer[:answer.find('\n')]
    return r
    # return answer

@app.route('/post_openai_conv_sum', methods=['POST'])
def post_openai_conv_sum():
    data = request.form
    question = data['question']
    response = summary_chain.run(question)
    r = response
    return r
    # return answer

@app.route('/get_app', methods=['GET', 'POST'])
def get_app():
    # uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    path = 'app.tar.gz'
    return send_file(path, as_attachment=True)



# app.run(host='0.0.0.0', port=9888,debug=True) # if we set debug to True, it will load the model twice, which will explode the GPU memory
app.run(host='0.0.0.0', port=9888, debug=False)

# Host on ipv6
# app.run(host='::', port=9888,debug=True)