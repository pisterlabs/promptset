
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load the vector database
model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

vectordb = Chroma(persist_directory='db/vectordb.db', embedding_function=embedding)

retriever = vectordb.as_retriever()

def create_prompt_from_RAG(user_input):
    retrieved_documents = retriever.get_relevant_documents(user_input)
    prompt = "The following are tweets with their respective classifications (Disaster or Not Disaster):\n"
    for doc in retrieved_documents:
        content = doc.page_content
        target = "Disaster" if doc.metadata['target'] == 1 else "Not Disaster"
        prompt += f"- Tweet: \"{content}\" Classification: {target}\n"
    prompt += "\nBased on the above tweets, provide your Classification, Justification of whether the tweet is related to Any Kind of Disaster or Not, by ignoring tone and other intricacies in human emotion and just focus on the logic\n\n"
    prompt += f"\"{user_input}\"\n\n"
    prompt += "Classification:"
    return prompt

if __name__ == "__main__":
    print(create_prompt_from_RAG("I love my life"))