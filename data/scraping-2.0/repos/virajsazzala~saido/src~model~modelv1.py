import sys
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


def model(data, query):
    # Configuration
    repo_id = "mistralai/Mistral-7B-v0.1"

    try:
        llm = HuggingFaceHub(
            huggingfacehub_api_token="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",
            repo_id=repo_id,
            model_kwargs={"temperature": 0.4, "max_new_tokens": 100},
        )
    except Exception as e:
        print(f"Error: Failed to initialize HuggingFaceHub. {str(e)}")
        sys.exit(1)

    for text in data:
        embeddings = HuggingFaceEmbeddings()

        db = Chroma.from_texts(text, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever, return_source_documents=True
        )

        chat_history = []

        result = qa_chain({"question": query, "chat_history": chat_history})

        answer = result["answer"].split("\n\nQuestion:")[0].strip()
        start_index = answer.find("\n\nHelpful Answer:")
        answer = answer[:start_index].strip()

        chat_history.append((query, answer))

    return answer


if __name__ == "__main__":
    with open(
        "/home/exvynai/code/dev/saido/src/model/transcriptions.json", "r"
    ) as json_file:
        data = json.load(json_file)

    while True:
        query = input("Prompt: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting")
            sys.exit()

        res = model(data, query)
        print(res)
