from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS


def prepare_texts():
    """テキストデータの準備"""
    return [
        "Pythonについての基本情報",
        "機械学習の最新トレンド",
        "データサイエンスの応用例"
    ]

def create_vectorstore(texts):
    """VectorStoreの作成"""
    return FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

def create_prompt():
    """プロンプトの作成"""
    return ChatPromptTemplate.from_template(
        "以下のcontextだけに基づいて回答してください。\n{context}\n質問: {question}"
    )

def create_model():
    """モデルの作成"""
    return ChatOpenAI(model_name="gpt-3.5-turbo")

def create_chain(retriever, prompt, model):
    """LCELでのチェーンの定義"""
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | model

def main():
    texts = prepare_texts()
    vectorstore = create_vectorstore(texts)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    prompt = create_prompt()
    model = create_model()
    chain = create_chain(retriever, prompt, model)

    question = "Pythonの基本的な特徴は何ですか？"
    result = chain.invoke(question)
    print(result)

if __name__ == "__main__":
    main()
