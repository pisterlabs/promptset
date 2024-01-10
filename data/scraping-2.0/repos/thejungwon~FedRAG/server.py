import os
import argparse
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


@app.route("/")
def index():
    """
    Serve the main page with a simple form to submit a question.
    """
    return """
        <form action="/ask" method="post">
            Ask a question: <input type="text" name="question">
            <input type="submit" value="Ask">
        </form>
    """


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Receive the user question, process it and respond with an answer.
    Utilizes the global `qa` instance for querying.
    """
    question = request.form.get("question")
    print(question)
    result = qa({"query": question})
    print(result["result"])

    return jsonify(
        {"user_id": user_id, "question": result["query"], "answer": result["result"]}
    )


def initialize_qa_chain(user_id):
    """
    Initialize the QA chain with specific documents, and return it.

    Parameters:
    - user_id: int
    """
    loaders = [
        PyPDFLoader("materials/Norway - Wikipedia.pdf"),
        PyPDFLoader("materials/Sweden - Wikipedia.pdf"),
        PyPDFLoader("materials/Denmark - Wikipedia.pdf"),
    ]

    loader = loaders[user_id]
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask app with specific user ID.")
    parser.add_argument(
        "--user_id",
        type=int,
        required=True,
        help="User ID to determine the port number.",
    )
    args = parser.parse_args()

    user_id = args.user_id
    qa = initialize_qa_chain(user_id)
    port = 6000 + args.user_id

    app.run(debug=True, host="0.0.0.0", port=port)
