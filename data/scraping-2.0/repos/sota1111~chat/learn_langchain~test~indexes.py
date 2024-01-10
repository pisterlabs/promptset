from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


# Document Loaderの使い方
loader = PyPDFLoader("https://blog.freelance-jp.org/wp-content/uploads/2023/03/FreelanceSurvey2023.pdf")
pages = loader.load_and_split()
print("pages:",pages[0])

chroma_index = Chroma.from_documents(pages, OpenAIEmbeddings())
#docs = chroma_index.similarity_search("「フリーランスのリモートワークの実態」について教えて。", k=2)
docs = chroma_index.similarity_search("「外部人材の利用」について教えて。", k=3)
for doc in docs:
    print("page:",str(doc.metadata["page"]) + ":", doc.page_content)

# Text Splittersの使い方
from langchain.text_splitter import CharacterTextSplitter
long_text = """
GPT-4は、OpenAIが開発したAI技術であるGPTシリーズの第4世代目のモデルです。

自然言語処理(NLP)という技術を使い、文章の生成や理解を行うことができます。

これにより、人間と同じような文章を作成することが可能です。

GPT-4は、トランスフォーマーアーキテクチャに基づいており、より強力な性能を発揮します。

GPT-4は、インターネット上の大量のテキストデータを学習し、豊富な知識を持っています。

しかし、2021年9月までの情報しか持っていません。

このモデルは、質問応答や文章生成、文章要約など、様々なタスクで使用できます。

ただし、GPT-4は完璧ではありません。

時々、誤った情報や不適切な内容を生成することがあります。

使用者は、その限界を理解し、

適切な方法で利用することが重要です。
"""
print("len(long_text):",len(long_text))

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = len,
)
text_list = text_splitter.split_text(long_text)
print("text_list:",text_list,"\n")
print("len(text_list)",len(text_list),"\n")

document_list = text_splitter.create_documents([long_text])
print("document_list:",document_list,"\n")
print("len(document_list)",len(document_list),"\n")