from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    lang="ja",
)
documents = retriever.get_relevant_documents(
    "大規模言語モデル"
)

print(f"検索結果： {len(documents)}件")

for document in documents:
    print("----------取得したメタデータ----------")
    print(document.metadata)
    print("----------取得したテキスト----------")
    print(document.page_content[:100])
