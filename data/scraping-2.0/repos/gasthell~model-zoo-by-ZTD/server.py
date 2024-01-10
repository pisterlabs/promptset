from flask import Flask, render_template, request, url_for, flash, redirect
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ZTD'

template = """You are ChatOSZ, a language model created by ZTD. You speak only on english. You help with writing research. Your main source of information is https://osdr.nasa.gov/. And you could search from another source if you couldnt find information
Human: {human_input}
ChatOSZ:"""

prompt = PromptTemplate(input_variables=["human_input"], template=template)

cllama_chain = LLMChain(
    llm="OSZ.bin",
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096}
)

messages = [""]
def qchat(input1):
    print("I got clicked")
    loader = WebBaseLoader("https://scholar.google.com/"+input1)
    docs = loader.load()
    chain = load_summarize_chain(cllama_chain, chain_type="stuff")
    chain.run(docs)
    output = cllama_chain.predict(input1)
    return(output)

@app.route('/answer',methods=['GET', 'POST'])
def answer():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            messages[0] = qchat(content)
            return redirect(url_for('answer'))
    print(messages)
    print(messages[0])
    return render_template('answer.html', messages=messages)

@app.route('/library',methods=['GET', 'POST'])
def library():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            messages[0] = qchat(content)
            return redirect(url_for('answer'))
    return render_template('library.html')

@app.route('/create',methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            messages[0] = qchat(content)
            return redirect(url_for('answer'))
    return render_template('create.html')

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            messages[0] = qchat(content)
            return redirect(url_for('answer'))
    return render_template('index.html')
  
if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)