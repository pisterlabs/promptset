from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Optional, Generator, List, Any

import pandas as pd
from asyncio_pool import AioPool
from tqdm import tqdm

from consts.api_consts import AIO_POOL_SIZE
from data_collection.openai.openai_client import OpenAIClient
from tools.data_chunks_generator import DataChunksGenerator
from utils.file_utils import append_to_csv


class BaseEmbeddingsCollector(ABC):
    def __init__(self,
                 output_path: str,
                 chunk_size: int = 50,
                 chunks_limit: Optional[int] = None,
                 openai_client: Optional[OpenAIClient] = None):
        self._output_path = output_path
        self._chunk_size = chunk_size
        self._chunks_limit = chunks_limit
        self._openai_client = openai_client
        self._data_chunks_generator = DataChunksGenerator(self._chunk_size, self._chunks_limit)

    @abstractmethod
    async def collect(self) -> None:
        raise NotImplementedError

    async def _extract_single_chunk_embeddings(self, chunk: list) -> None:
        records = await self._get_embeddings_records(chunk)
        data = pd.DataFrame.from_records(records)

        append_to_csv(data=data, output_path=self._output_path)

    async def _get_embeddings_records(self, chunk: list) -> List[dict]:
        pool = AioPool(AIO_POOL_SIZE)

        with tqdm(total=len(chunk)) as progress_bar:
            func = partial(self._extract_single_embeddings, progress_bar)
            records = await pool.map(func, chunk)

        return [record for record in records if isinstance(record, dict)]

    @abstractmethod
    async def _extract_single_embeddings(self, progress_bar: tqdm, unit: Any) -> List[dict]:
        raise NotImplementedError

    async def _get_embeddings_record(self, prompt: str) -> Dict[str, float]:
        embeddings = await self._openai_client.embeddings(prompt)
        return {f'lyrics_embedding_{i + 1}': embedding for i, embedding in enumerate(embeddings)}

    async def __aenter__(self) -> 'BaseEmbeddingsCollector':
        self._openai_client = await OpenAIClient().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._openai_client:
            await self._openai_client.__aexit__(exc_type, exc_val, exc_tb)
