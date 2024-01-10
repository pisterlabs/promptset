import db
import json
from flask import Flask, request, render_template
from flask import jsonify
# import model1

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA


from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

global qa_result
qa_result = None

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )    
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    response = qa_result({'query': query}) 
    # Calling qa_bot function
    return response

# global qa_result
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template("signin.html")


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    return render_template("signup.html")



@app.route('/signin', methods = ['GET', 'POST'])
def signin():
    status, username = db.check_user()

    data = {
        "username": username,
        "status": status
    }

    return json.dumps(data)



@app.route('/register', methods = ['GET', 'POST'])
def register():
    status = db.insert_data()
    return json.dumps(status)


def query():
	if request.method == 'POST':
		# query = request.form['query']
		request_data = request.get_json()
		print("QUERY")
		print(request_data['query'])
		response = final_result(request_data['query'])
		# print(response['result'])
		documents = response["source_documents"]
		extracted_data = []

	for document in documents:	
		extracted_data.append({
			'content': document.page_content,
			'source': document.metadata['source'],
			'page': document.metadata['page']
		})
	# Convert the extracted data to JSON
	json_data = json.dumps(extracted_data, indent=4)
	ans ={
			"query": request_data['query'],
			"result": response['result'],
			"metadata":json_data
		}
	# save ans in db match it with email ?
	return ans

# TODO: STILL INCOLMPLETE 
@app.route('/ask', methods = ['GET', 'POST'])
def ask():
    status = query()
    print(status)
    return jsonify(status)

if __name__ == '__main__':
    qa_result = qa_bot()
    app.run(host= '0.0.0.0',debug=True)