from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma

# --------------------------------------------------
# プレーンテキストの読み込み
# 読み込んだテキストのチャンキング(分割)
# --------------------------------------------------

# テキストローダインスタンスを作成
loader = TextLoader("./load/data/intro.txt")

# テキストローダインスタンスメソッドを使用して読み込み
document = loader.load()

# テキストをチャンクごとに分割するインスタンスを作成
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

# テキストをチャンキング
documents = text_splitter.create_documents([document[0].page_content])

# --------------------------------------------------
# おまけ
# 読み込んだ Document を chroma DB に格納
# その後類似度検索
# --------------------------------------------------

# GPT4All モデルのパスを指定
model_path = "/mnt/c/Users/46ki7/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"

# GPT4All インスタンスを作成
llm = GPT4All(model=model_path)

# ベクトル表現(埋め込み)を生成するインスタンスを作成
embeddings = GPT4AllEmbeddings(llm=llm)

# chroma DBインスタンスを作成
db: Chroma = Chroma.from_documents(documents, embeddings)

# 検索クエリ
query: str = "What is feature of langchain?"

# 埋め込みモデルを使用して検索クエリを実行
embedding_vector: [float] = GPT4AllEmbeddings(llm=llm).embed_query(query)

# ベクターストア検索
results = db.similarity_search_by_vector(embedding_vector)

# 結果を標準出力に印字
for index, result in enumerate(results):
    print(f"# simularity {index} --------------------")
    print(result.page_content, "\n")

# 参考: チャンキングした documents の中身
"""
documents = [
    Document(page_content='Introduction', metadata={'start_index': 0}),
    Document(page_content='LangChain is a framework for developing applications powered by language models. It enables',
             metadata={'start_index': 13}),
    Document(page_content='models. It enables applications that:',
             metadata={'start_index': 86}),
    Document(page_content='Are context-aware: connect a language model to sources of context (prompt instructions, few shot',
             metadata={'start_index': 125}),
    Document(page_content='few shot examples, content to ground its response in, etc.)',
             metadata={'start_index': 213}),
    Document(page_content='Reason: rely on a language model to reason (about how to answer based on provided context, what',
             metadata={'start_index': 273}),
    Document(page_content='context, what actions to take, etc.)',
             metadata={'start_index': 355}),
    Document(page_content='The main value props of LangChain are:',
             metadata={'start_index': 392}),
    Document(page_content='Components: abstractions for working with language models, along with a collection of',
             metadata={'start_index': 432}),
    Document(page_content='a collection of implementations for each abstraction. Components are modular and easy-to-use,',
             metadata={'start_index': 502}),
    Document(page_content='and easy-to-use, whether you are using the rest of the LangChain framework or not',
             metadata={'start_index': 579}),
    Document(page_content='Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level',
             metadata={'start_index': 661}),
    Document(page_content='higher-level tasks', metadata={'start_index': 746}),
    Document(page_content='Off-the-shelf chains make it easy to get started. For complex applications, components make it easy',
             metadata={'start_index': 765}),
    Document(page_content='make it easy to customize existing chains and build new ones.',
             metadata={'start_index': 852}),
    Document(page_content='Get started\nHere’s how to install LangChain, set up your environment, and start building.',
             metadata={'start_index': 915}),
    Document(page_content='We recommend following our Quickstart guide to familiarize yourself with the framework by building',
             metadata={'start_index': 1006}),
    Document(page_content='by building your first LangChain application.',
             metadata={'start_index': 1093}),
    Document(page_content='Note: These docs are for the LangChain Python package. ForDocumentation on LangChain.js, the JS/TS',
             metadata={'start_index': 1140}),
    Document(page_content='the JS/TS version, head here.',
             metadata={'start_index': 1230}),
    Document(page_content='Modules', metadata={'start_index': 1261}),
    Document(page_content='LangChain provides standard, extendable interfaces and external integrations for the following',
             metadata={'start_index': 1269}),
    Document(page_content='for the following modules, listed from least to most complex:',
             metadata={'start_index': 1346}),
    Document(page_content='Model I/O\nInterface with language models\n\nRetrieval\nInterface with application-specific data',
             metadata={'start_index': 1409}),
    Document(page_content='Chains\nConstruct sequences of calls',
             metadata={'start_index': 1503}),
    Document(page_content='Agents\nLet chains choose which tools to use given high-level directives',
             metadata={'start_index': 1540}),
    Document(page_content='Memory\nPersist application state between runs of a chain',
             metadata={'start_index': 1613}),
    Document(page_content='Callbacks\nLog and stream intermediate steps of any chain',
             metadata={'start_index': 1671}),
    Document(page_content='Examples, ecosystem, and resources\nUse cases',
             metadata={'start_index': 1729}),
    Document(page_content='Use cases\nWalkthroughs and best-practices for common end-to-end use cases, like:',
             metadata={'start_index': 1764}),
    Document(page_content='Document question answering\nChatbots\nAnalyzing structured data\nand much more...\nGuides',
             metadata={'start_index': 1846}),
    Document(page_content='Guides\nLearn best practices for developing with LangChain.',
             metadata={'start_index': 1926}),
    Document(page_content='Ecosystem', metadata={'start_index': 1986}),
    Document(page_content='LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top',
             metadata={'start_index': 1996}),
    Document(page_content='and build on top of it. Check out our growing list of integrations and dependent repos.',
             metadata={'start_index': 2077}),
    Document(page_content='Additional resources',
             metadata={'start_index': 2166}),
    Document(page_content='Our community is full of prolific developers, creative builders, and fantastic teachers. Check out',
             metadata={'start_index': 2187}),
    Document(page_content='teachers. Check out YouTube tutorials for great tutorials from folks in the community, and Gallery',
             metadata={'start_index': 2266}),
    Document(page_content='and Gallery for a list of awesome LangChain projects, compiled by the folks at KyroLabs.',
             metadata={'start_index': 2353}),
    Document(page_content='Community', metadata={'start_index': 2443}),
    Document(page_content='Head to the Community navigator to find places to ask questions, share feedback, meet other',
             metadata={'start_index': 2453}),
    Document(page_content='meet other developers, and dream about the future of LLM’s.',
             metadata={'start_index': 2534}),
    Document(page_content='API reference', metadata={'start_index': 2595}),
    Document(page_content='Head to the reference section for fullDocumentation of all classes and methods in the LangChain',
             metadata={'start_index': 2609}),
    Document(page_content='in the LangChain Python package.',
             metadata={'start_index': 2689})
]
"""

# Document の構造は以下のようになる
"""
    {
  "page_content": "チャンクの内容",
  "metadata": {
    "start_index": 100 
  } 
}
"""
