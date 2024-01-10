from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def initialize_chat_bot():

    db = FAISS.load_local("QandAIndex", OpenAIEmbeddings())
    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    global topK_retriever
    topK_retriever = db.as_retriever(top_k=3)
    global chat_bot
    # chat_bot = RetrievalQA.from_chain_type(llm,
    #                                        retriever=db.as_retriever(search_type="similarity_score_threshold",
    #                                                                  search_kwargs={"score_threshold": 0.8}))
    chat_bot = RetrievalQA.from_chain_type(llm, retriever=topK_retriever)
    chat_bot.return_source_documents = True
# docs = topK_retriever.get_relevant_documents("What is the smart choice?")

# for doc in docs:
#     print(doc)

def launch_gradio():
    import gradio as gr

    def chatbot(question):
        # docs = topK_retriever.get_relevant_documents(question)
        # return docs[0]
        ans = chat_bot(question)
        
        return ans["result"]
        

    iface = gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="Smart Choice Chatbot",
        description="Chatbot that answers questions about Smart Choice.",
        examples=[
            ["What is the smart choice?"]
        ]
    )

    iface.launch()


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_chat_bot()
    # 启动 Gradio 服务
    launch_gradio()