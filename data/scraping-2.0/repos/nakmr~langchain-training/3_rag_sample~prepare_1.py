from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

print(f"ドキュメントの数: {len(documents)}")

print(f"１つ目のドキュメントの内容: {documents[0].page_content}")
print(f"１つ目のドキュメントのメタデータ: {documents[0].metadata}")