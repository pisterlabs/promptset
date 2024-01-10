from typing import Optional, Union

from langchain.schema import Document
from pydantic import BaseModel


class DocumentChunker(BaseModel):
    chunk_size: Optional[Union[int, None]]
    chunk_overlap: Optional[Union[int, None]]
    documents: Union[str, list[Document]]

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "documents": [
                    {
                        "page_content": "{\"userId\": 1, \"id\": 1, \"title\": \"delectus aut autem\", \"completed\": "
                                        "false}",
                        "metadata": {
                            "source": "/private/var/folders/pm/5v47_r592xz7d9rm1jb53tyh0000gn/T/tmprszk62ep",
                            "seq_num": 1
                        }
                    }
                ]
            }
        }


class VectorStore(BaseModel):
    host: str

    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost"
            }
        }


class StoreInVectoDB(BaseModel):
    vectorstore: Optional[str] = "chromadb"
    embedding_model: Optional[str] = "openai"
    documents: list[Document]

    class Config:
        json_schema_extra = {
            "example": {
                "vectorstore": "chromadb",
                "embedding_model": "openai",
                "documents": [
                        {
                            "page_content": "Some payload that is chunked into multiple smaller documents",
                            "metadata": {
                                "source": "/private/var/folders/pm/5v47_r592xz7d9rm1jb53tyh0000gn/T/tmphr547jgv",
                                "seq_num": 1
                            }
                        }
                ]
            }
        }

