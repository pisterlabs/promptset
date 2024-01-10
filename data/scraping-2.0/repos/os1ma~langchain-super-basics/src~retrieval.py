from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS

load_dotenv()

# 文書をベクトル化してVector Storeに格納して検索準備
texts = [
    "私の趣味は読書です。",
    "私の好きな食べ物はカレーです。",
    "私の嫌いな食べ物は饅頭です。",
]
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 「検索 => プロンプトの穴埋め => LLMで回答を生成」というchainを作成
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# chainを実行
result = chain.invoke("私の好きな食べ物はなんでしょう？")
print(result.content)
