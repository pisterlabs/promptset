#%%
from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from glob import glob
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv('.env')

from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQAWithSourcesChain
#%%

loader = DirectoryLoader("./converted_texts",glob = "*.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
db = Chroma.from_documents(documents, OpenAIEmbeddings())
llm = OpenAI(temperature=0)


# chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_type="mmr"), memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))

#%%

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=db.as_retriever())#, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))
    response = qa_chain({"question": prompt})
    # Merge 'answer' and 'sources' keys into a single dictionary
    memory_output = {'response': response['answer'], 'sources': response['sources']}
    
    # Save the context with the merged dictionary as output
    # qa_chain.memory.save_context({"question": prompt}, memory_output)
    
    return jsonify(memory_output)





@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"hello": "world"})


if __name__ == "__main__":
    app.run(debug=True)
# %%


