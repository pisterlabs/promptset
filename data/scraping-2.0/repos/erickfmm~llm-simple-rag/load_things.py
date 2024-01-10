from langchain.embeddings import HuggingFaceEmbeddings

if __name__ == "__main__":
	HuggingFaceEmbeddings(model_name='dccuchile/bert-base-spanish-wwm-uncased')