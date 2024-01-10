from uuid import uuid4

import pinecone
import tiktoken
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from app.config import pinecone_settings

BATCH_SIZE = 100


class TiktokenClient:
    def __init__(self):
        tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_encoding(self, text):
        return self.tokenizer.encode(text, disallowed_special=())

    def tiktoken_len(self, text):
        tokens = self._get_encoding(text)
        return len(tokens)


tiktoken_client = TiktokenClient()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_client.tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)


class PineconeClient:
    def __init__(self):
        pinecone.init(
            api_key=pinecone_settings.api_key, environment=pinecone_settings.environment
        )
        self.index = pinecone.Index(pinecone_settings.index)

    def _download_dataset(self):
        data = load_dataset("wikipedia", "20220301.simple", split="train[:10000]")
        return data

    def get_chunks(self, data):
        chunks = text_splitter.split_text(data[6]["text"])[:3]
        return chunks

    def populate_index(self, embed: OpenAIEmbeddings):
        texts = []
        metadatas = []
        data = self._download_dataset()
        # chunks = self.get_chunks(data)
        for _i, record in enumerate(tqdm(data)):
            # first get metadata fields
            metadata = {
                "wiki-id": str(record["id"]),
                "source": record["url"],
                "title": record["title"],
            }
            # now we create chunks from the record text
            record_texts = text_splitter.split_text(record["text"])
            # create individual metadata dicts for each chunk
            record_metadatas = [
                {"chunk": j, "text": text, **metadata}
                for j, text in enumerate(record_texts)
            ]
            # append these to current batches
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if we have reached the batch_limit we can add texts
            if len(texts) >= BATCH_SIZE:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = embed.embed_documents(texts)
                self.index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []
        print(self.index.describe_index_stats())
        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            self.index.upsert(vectors=zip(ids, embeds, metadatas))
        print(self.index.describe_index_stats())


pinecone_client = PineconeClient()
