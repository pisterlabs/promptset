from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.vectorstores.weaviate import Weaviate
import weaviate
import os


if __name__ == "__main__":
    client = weaviate.Client("http://localhost:8080")
    vs = Weaviate(client, "Document", "text")

    MyOpenAI = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa = ChatVectorDBChain.from_llm(MyOpenAI, vs)

    chat_history = []

    ongoing = True

    while ongoing:
        print("=============================================")
        print("Please enter a question or dialogue to get started or type 'Abort' to end the conversation!")  # noqa: E501

        query = input("")
        if query == "Abort":
            ongoing = False
            break
        result = qa({"question": query, "chat_history": chat_history})
        print(result["answer"])
        chat_history = [(query, result["answer"])]
