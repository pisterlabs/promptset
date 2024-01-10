from flask import Flask, request, jsonify
from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.schema.prompt_template import format_document
from langchain.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from dotenv import load_dotenv
import logging
import os

load_dotenv()

app = Flask(__name__)

# Log to app.log
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
                    filename='app.log',
                    filemode='a')


def include_file(file_path):
    include_extensions = ['.txt', '.md', '.yml', '.yaml',
                          '.env', '.js', '.jsx', '.html', '.css', '.py', '.sh']
    include_names = ['Dockerfile', 'docker-compose.yml']

    return (
        any(file_path.endswith(ext) for ext in include_extensions) or
        any(file_path.endswith(name) for name in include_names)
    )


# Define GitLoader with the specified repository and filter for .js files
loader = GitLoader(
    clone_url="https://github.com/TriumvirateTechnologySolutions/AutoDoc",
    repo_path="./data/",
    branch="main",
    file_filter=include_file,
)
documents = loader.load()
print(f"Loaded {len(documents)} documents on branch main.")

SIMILARITY_K = 4
chunk_size_value = 2000
chunk_overlap = 200
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size_value, chunk_overlap=chunk_overlap, length_function=len)
texts = text_splitter.split_documents(documents)
docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
docembeddings.save_local("llm_faiss_index")
docembeddings = FAISS.load_local("llm_faiss_index", OpenAIEmbeddings())

map_prompt = PromptTemplate.from_template(
    "Answer the user question using the context. Place any code in ```code here``` blocks."
    "\n\nContext:\n\n{context}\n\nQuestion: {question}"
)


class AnswerAndScore(BaseModel):
    """A model to hold the answer to a question and a corresponding relevance score."""

    answer: str = Field(
        description="The answer to the question, based ONLY on the provided context.")
    score: float = Field(
        description="A 0.0-1.0 relevance score, where 1.0 indicates the provided context answers the question completely and 0.0 indicates the provided context does not answer the question at all.")


function = convert_pydantic_to_openai_function(AnswerAndScore)

map_chain = (
    map_prompt
    | ChatOpenAI().bind(
        temperature=0, functions=[function], function_call={"name": "AnswerAndScore"}
    )
    | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
).with_config(run_name="Map")

document_prompt = PromptTemplate.from_template("{page_content}")


def top_answer(scored_answers):
    for idx, scored_answer in enumerate(scored_answers):
        logging.info(f"Document {idx + 1} Score: {scored_answer.score}")
    return max(scored_answers, key=lambda x: x.score).answer


map_rerank_chain = (
    (
        lambda x: [
            {
                "context": format_document(doc, document_prompt),
                "question": x["question"],
            }
            for doc in x["docs"]
        ]
    )
    | map_chain.map()
    | top_answer
).with_config(run_name="Map rerank")


def getanswer(query):
    relevant_chunks = docembeddings.similarity_search_with_score(query, k=SIMILARITY_K)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]
    results = map_rerank_chain.invoke({"docs": chunk_docs, "question": query})
    text_reference = "\n\n".join(doc.page_content for doc in chunk_docs)
    output = {"Answer": results, "Reference": text_reference}
    return output


@app.route('/reporag', methods=["POST"])
def processclaim():
    try:
        input_json = request.get_json(force=True)
        query = input_json["query"]
        output = getanswer(query)
        return output
    except:
        return jsonify({"Status": "Failure --- some error occured"})


if __name__ == "__main__":
    print("Flask app running...")
    app.run(host="0.0.0.0", port=6000, debug=False)
