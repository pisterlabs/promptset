import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import openai


class VectorDBService:
    def __init__(self,
                 environment="",
                 device='cpu',
                 model='bert-base-uncased',
                 index_name=" ",
                 batch_size=256,
                 provider_API_key = "",
                 vectordb_API_key = ""
                 ):


        if provider_API_key == "":
            print("provider API key is not set")
            return

        self.provider_API_key = provider_API_key
        self.vectordb_API_key = vectordb_API_key
        self.env = environment
        self.batch_size = batch_size
        self.model = SentenceTransformer(model, device=device)

        pinecone.init(
            api_key=self.vectordb_API_key,
            environment=self.env
        )

        self.index = pinecone.Index(index_name)

        print("correctly initialized vector database")
    def batch_words(self, data: list, seq_len=256):

        batches = []
        for i in range(0, len(data), seq_len):
           end = min(i+seq_len, len(data))
           batches.append(" ".join(data[i:end]))
        return batches


    def vectorize_and_upsert(self, data: list):
        string = " ".join(data)
        string = " ".join(string.split())
        words = string.split(" ")

        batch_words_256_seq_length = self.batch_words(words)

        for i in tqdm(range(0, len(batch_words_256_seq_length), self.batch_size)):
            end = i + self.batch_size
            ids = [str(x) for x in range(i, end)]
            metadata = [{'text' : text} for text in batch_words_256_seq_length[i:end]]
            xc = self.model.encode(batch_words_256_seq_length[i:end]).tolist()
            records = zip(ids, xc, metadata)
            self.index.upsert(vectors=records)
            print(f'uploaded records {ids}')

    def query(self, query : str, top_k=5):
        print("performing RAG query...")
        xc = self.model.encode(query).tolist()
        return self.index.query(xc, top_k=top_k, include_metadata=True)

    def run_conversation(self, query, init: dict):

        openai.api_key = self.provider_API_key

        RAG = self.query(query)

        messages = []
        messages.append(init)

        for i in range(0, len(RAG["matches"])):
            msg = {"role": "system", "content" : RAG["matches"][i]["metadata"]["text"]}
            messages.append(msg)

        messages.append({"role": "user", "content" : query})

        try:
            print("Performing RAG chat completion...")
            chat_completion = openai.ChatCompletion.create(
                model='gpt-4',
                messages=messages
            )
            return chat_completion

        except Exception as e:
            print(e)


