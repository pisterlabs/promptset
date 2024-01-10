from typing import List, Optional
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import openai
from tqdm.auto import tqdm
import os
from pod_gpt.models import Record, VideoRecord, Metadata

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


class Chunker:
    def __init__(self, chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # number of tokens overlap between chunks
            length_function=tiktoken_len,
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )

    def __call__(self, video_record: VideoRecord) -> List[Record]:
        text_chunks = self.text_splitter.split_text(video_record.transcript)
        return [
            Record(
                id=f'{video_record.video_id}-{i}',
                text=text,
                metadata=Metadata(
                    title=video_record.title,
                    channel_id=video_record.channel_id,
                    published=video_record.published,
                    source=video_record.source,
                    chunk=i
                )
            )
            for i, text in enumerate(text_chunks)
        ]


class Indexer:
    dimension_map = {
        'text-embedding-ada-002': 1536
    }
    def __init__(
        self, openai_api_key: Optional[str], pinecone_api_key: Optional[str],
        pinecone_environment: Optional[str], index_name: Optional[str] = "pod-gpt",
        embedding_model_name: Optional[str] = "text-embedding-ada-002",
        chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20
    ):
        # get variables
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        self.pinecone_environment = pinecone_environment or os.environ.get('PINECONE_ENVIRONMENT')
        if self.openai_api_key is None:
            raise ValueError('openai_api_key not specified')
        if self.pinecone_api_key is None:
            raise ValueError('pinecone_api_key not specified')
        if self.pinecone_environment is None:
            raise ValueError('pinecone_environment not specified')
        
        # initialize everything else...
        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedding_model_name = embedding_model_name
        self.metadata_config = {'indexed': list(Metadata.schema()['properties'].keys())}
        # initialize pinecone connection
        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment
        )
        # initialize openai connection
        openai.api_key = openai_api_key
        # initialize pinecone index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name, dimension=self.dimension_map[embedding_model_name],
                metadata_config=self.metadata_config
            )
        # connect to Pinecone index
        self.index = pinecone.GRPCIndex(index_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        res = openai.Embedding.create(
            input=texts,
            engine=self.embedding_model_name
        )
        return [result["embedding"] for result in res["data"]]

    def _index(self, records: List[Record]) -> None:
        ids = [record.id for record in records]
        texts = [record.text for record in records]
        metadatas = [dict(record.metadata) for record in records]
        # add texts to metadata
        for i, metadata in enumerate(metadatas):
            metadata['text'] = texts[i]
        embeddings = self._embed(texts)
        # upsert to Pinecone index
        self.index.upsert(vectors=zip(ids, embeddings, metadatas))
    
    def __call__(self, video_record: VideoRecord, batch_size: Optional[int] = 100) -> None:
        chunks = self.chunker(video_record)
        for i in range(0, len(chunks), batch_size):
            i_end = min(i + batch_size, len(chunks))
            self._index(chunks[i:i_end])
