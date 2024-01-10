from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import faiss
import numpy as np
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

index = faiss.read_index('/Users/amitabhranjan/IdeaProjects/PDFChatbot/ChatBotlangchain/faiss_index.bin')
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.2, "max_length": 256}
)
chain = load_qa_chain(llm, chain_type="stuff")
# query = "when was cristiano ronaldo born ?"
query = "when did cristiano ronaldo played his first official match in a UEFA Champions League "
query_vector = np.array([query])  # Convert query to 2D numpy array
query_vector = np.reshape(query_vector, (1, -1))  # Reshape query_vector to have 2 dimensions
_, I = index.search(query_vector, k=5)  # Perform similarity search
docs = [index.reconstruct(i) for i in I[0]]
res = chain.run(input_documents=docs, question=query)
print(res)