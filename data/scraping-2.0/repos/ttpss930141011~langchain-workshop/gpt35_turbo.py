from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 加載 youtube 頻道
loader = YoutubeLoader.from_youtube_url(
    'https://www.youtube.com/watch?v=Dj60HHy-Kqk')
# 將數據轉成 document
documents = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

# 分割 youtube documents
documents = text_splitter.split_documents(documents)

# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()

# 將數據存入向量存儲
vector_store = Chroma.from_documents(documents, embeddings)
# 通過向量存儲初始化檢索器
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{question}
-----------
{chat_history}
"""

# 構建初始 messages 列表，這裡可以理解為是 openai 傳入的 messages 參數
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 對象
prompt = ChatPromptTemplate.from_messages(messages)


# 初始化問答鏈
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(
    temperature=0.1, max_tokens=2048), retriever, condense_question_prompt=prompt)


chat_history = []
while True:
    question = input('問題：')
    # 開始發送問題 chat_history 為必須參數,用於存儲對話歷史
    result = qa({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    print(result['answer'])