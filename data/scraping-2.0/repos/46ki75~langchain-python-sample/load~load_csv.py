from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma

# --------------------------------------------------
# CSV の読み込み
# 読み込んだ CSV のチャンキング(分割)
# --------------------------------------------------

loader = CSVLoader(file_path='./load/data/jpholiday.csv')
data = loader.load()
print(data)

# --------------------------------------------------
# おまけ
# 読み込んだ Document を chroma DB に格納
# その後類似度検索
# --------------------------------------------------

# GPT4Allモデルのパスを指定
model_path = "/mnt/c/Users/46ki7/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"

# GPT4Allインスタンスを作成
llm = GPT4All(model=model_path)

# ベクトル表現(埋め込み)を生成するインスタンスを作成
embeddings = GPT4AllEmbeddings(llm=llm)

# chroma DBインスタンスを作成
db: Chroma = Chroma.from_documents(data, embeddings)

# 検索クエリ
query: str = "What day is 2019-01-14?"

# 埋め込みモデルを使用して検索クエリを実行
embedding_vector: [float] = GPT4AllEmbeddings(llm=llm).embed_query(query)

# ベクターストア検索
results = db.similarity_search_by_vector(embedding_vector)

# 結果を標準出力に印字
for index, result in enumerate(results):
    print(f"# simularity {index} --------------------")
    print(result.page_content, "\n")
