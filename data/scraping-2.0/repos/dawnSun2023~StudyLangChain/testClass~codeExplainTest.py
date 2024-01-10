#文本嵌入模型
from langchain.embeddings.openai import OpenAIEmbeddings

"""
DeepLake是一个向量存储库，用于存储和检索向量
文档网址:https://docs.activeloop.ai/
官网地址：https://app.activeloop.ai/
"""
from langchain.vectorstores import DeepLake
#文档加载器用于从文件系统加载文档
from langchain.document_loaders import TextLoader
#文本分割器用于将文档分割为块
from langchain.text_splitter import CharacterTextSplitter, PythonCodeTextSplitter
#导入openAi模型
from langchain.chat_models import ChatOpenAI
"""
用于根据检索到的文档进行对话的链。
该链接收聊天历史记录（消息列表）和新问题，然后返回该问题的答案。该链的算法由三部分组成：
1. 使用聊天记录和新问题创建“独立问题”。这样做是为了将该问题传递到检索​​步骤以获取相关文档。如果只传入新问题，则可能缺乏相关上下文。如果将整个对话传递给检索，则可能会存在不必要的信息，从而分散检索的注意力。
2. 这个新问题被传递给检索器并返回相关文档。
3. 检索到的文档与新问题（默认行为）或原始问题和聊天历史记录一起传递给 LLM，以生成最终响应。
"""
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
#设置openAI的key 和 activeLoop的token
os.environ['OPENAI_API_KEY'] = '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'
#ActiveLoop的token
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4ODg3ODc1NywiZXhwIjoxNjg4OTY1MTM0fQ.eyJpZCI6ImNmNTY4Njc3Njg3In0.qX7EnkX6yr4Q2TI5CMe6T17lhHiWtlEPX_rEcgK147EyqYo7wMImhlNRQooNvlYjrMmT1cwHyQnGvhXi73gkJw'

sources = []
loader = TextLoader('../llm_base.py', encoding='utf8')
#拆分文档并将其添加到源列表中
sources.extend(loader.load_and_split())

#chunk_size=1000：指定每个拆分块的大小为1000个字符。这意味着源文本将被分成多个长度为1000字符的块。
#chunk_overlap=0：指定拆分块之间的重叠量为0。这意味着拆分块之间没有重叠。
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#将拆分的文档存储在docs中
docs = splitter.split_documents(sources)

# dataset_path = 'hub://dalio/langchain_code'
dataset_path = 'hub://cf568677687/text_embedding'
embeddings = OpenAIEmbeddings()

#它通过将文档数据和嵌入向量加载到指定的数据集路径中创建。用于后续的文档检索、相似性匹配等操作。
#这个过程会执行嵌入，并把结果上传到DeepLake平台，如果DeepLake中已经存在数据集，会跳过嵌入，自动从存储中加载。
db = DeepLake.from_documents(docs, embeddings, dataset_path=dataset_path)

#创建一个检索器，它将使用嵌入向量来检索文档
retriever = db.as_retriever()
#设置检索器的参数
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

#创建一个聊天模型，它将使用OpenAI模型来生成答案
model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0) # 'gpt-3.5-turbo',

#创建一个检索问答链，它将使用检索器和聊天模型来生成答案
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)


questions = [
    '解释这段代码'
]
chat_history = []

#对每个问题进行问答
for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")