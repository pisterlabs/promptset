from django.test import TestCase
import pinecone
import PathFinder.PathFinderModels.pathfinder_chat_bot as qamodel
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


def _test_qachain(request):
    # qamodel.load_embed_pickle()
    pinecone.init(
        api_key="5bf2927b-0fb7-423b-b8f1-2f6a3347a15d",
        environment="asia-northeast1-gcp",
    )
    vectorstore = Pinecone.from_existing_index("teamprojindex", OpenAIEmbeddings())
    pathfinder_chatbot = qamodel.make_chain(vectorstore)
    chat_history = []
    # print(vectorstore.similarity_search("what is a computer", 10))
    question = input("Enter a question: ")

    # answer = pathfinder_chatbot({
    #     "question": question,
    #     "chat_history": []
    # })
    # print(answer)
    # print(chatbot.)
    while 1:
        answer = pathfinder_chatbot(
            {"question": question, "chat_history": chat_history}
        )
        print()
        print(answer)
        chat_history.append({"role": "assistant", "content": answer})
        print("type exit to exit...")
        question = input()
        if question == "exit":
            break

    # with open("ndtmvecstore.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)
    #     print(qamodel.make_chain_prebuilt(vectorstore))

    return 0


# _test_qachain(None)
