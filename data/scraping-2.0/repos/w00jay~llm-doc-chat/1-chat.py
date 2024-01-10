import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
# pip install sentence-transformers


# If using BERT-based LLM:
# def generate_query_embedding(query, tokenizer, model):
    # inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # return outputs.last_hidden_state.mean(dim=1).numpy()  # Convert to numpy array

def generate_query_embedding(query, embedding_model):
    return embedding_model.encode([query])[0]

# Query ChromaDB to find the most relevant documents
def find_relevant_document(db, query):
    results = db.similarity_search(query)
    return results

# # Initialize tokenizer and model for generating query embeddings w/ BERT
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# model = AutoModel.from_pretrained("bert-large-uncased")

# Initialize the Sentence Transformer model
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

# with open("doc_info_mapping.json", "r") as file:
#     doc_info_mapping = json.load(file)

# Initialize the embedding and chroma
embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")  # "all-MiniLM-L6-v2"
chroma = Chroma(
    persist_directory="./test/chroma-all-mpnet-base-v2",
    embedding_function = embedding_function,
)

# Interactive chat loop
while True:
    user_input = input("Enter your query (or type 'q' to exit): ")
    if user_input.lower() == 'q':
        break

    # Generate query embedding
    # query_embedding = generate_query_embedding(user_input, tokenizer, model)
    query_embedding = generate_query_embedding(user_input, sentence_transformer_model)

    # Find the most relevant document
    relevant_docs = find_relevant_document(chroma, user_input)
    # print(f"Most relevant document index: {relevant_docs}")

    for doc in relevant_docs:
        print(doc.metadata)
        print(doc.page_content)
        print()
        
    # # Retrieve additional info based on the index
    # doc_info = doc_info_mapping[str(relevant_doc_index)]
    # print(f"Most relevant document: {doc_info}")

# End of the chat loop
print("Chat session ended.")
