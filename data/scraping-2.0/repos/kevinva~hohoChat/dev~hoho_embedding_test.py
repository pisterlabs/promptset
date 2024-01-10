from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import BertModel

path = "/root/hoho/models/embeddings/ms"

embedding = HuggingFaceEmbeddings(model_name = path)

# model = BertModel.from_pretrained(path)
# embedding = model.embeddings.word_embeddings

print(f'embeddig: {embedding}')