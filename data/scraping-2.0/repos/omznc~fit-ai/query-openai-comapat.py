from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from openai import OpenAI

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory="./chroma_openai_compat", embedding_function=embeddings)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="potrebo-samo-za-official-openai-api")  # Potreban openai kompatibilan server

while True:
	query = input("\nEnter query: ")
	if query == "exit":
		break
	results = db.similarity_search_with_relevance_scores(query, top_k=6)
	if len(results) == 0:
		print(f"Nista nije nadjeno za upit: {query}")
		continue

	context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

	stream = client.chat.completions.create(
		model="potrebo-samo-za-official-openai-api",
		messages=[
			{"role": "system","content": f"Always answer in the Bosnian language. Use only the provided context. Answer short and to the point. This is the context: {context}"},
			{"role": "user", "content": query},
		],
		temperature=0.7,
		stream=True,
		max_tokens=500,
	)

	for chunk in stream:
		print(chunk.choices[0].delta.content or "", end="")

	print(f"\nIzvori: {[doc[0].metadata.get('source', None) for doc in results]}")
