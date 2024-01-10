from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load the persisted database from disk
persist_directory = "db"
embedding = OpenAIEmbeddings()  # model = "text-embedding-ada-002"

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Make a retriever
retriever = vectordb.as_retriever()

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True
)


# Define the function to process and display results
def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


def display_llm_response(llm_response):
    print("Custom Result Processing:")
    print("Answer:", llm_response["result"])
    print("Sources:")
    for source in llm_response["source_documents"]:
        print("- Source:", source.metadata["source"])
        print("- Score:", source.score)


def format_llm_response(llm_response):
    result = llm_response["result"]
    sources = [source.metadata["source"] for source in llm_response["source_documents"]]

    # Format the response string
    response = f"Result: \n{result}\n\nSources:\n"
    for source in sources:
        response += f"- {source}\n"

    return response


def html_llm_response(llm_response):
    result = llm_response["result"]
    sources = [source.metadata["source"] for source in llm_response["source_documents"]]

    # Format the response string with HTML tags
    response = f"<h3>Result:</h3><p>{result}</p><h3>Sources:</h3>"
    for source in sources:
        response += f"<li>{source}</li>"

    return response
