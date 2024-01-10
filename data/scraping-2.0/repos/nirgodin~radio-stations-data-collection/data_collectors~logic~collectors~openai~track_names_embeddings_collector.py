from typing import List, Dict, Optional

from genie_common.models.openai import EmbeddingsModel
from genie_common.openai import OpenAIClient
from genie_common.tools import logger
from genie_common.utils import merge_dicts
from sqlalchemy.ext.asyncio import AsyncEngine

from data_collectors.contract import ICollector
from data_collectors.logic.models import MissingTrack
from genie_common.tools import AioPoolExecutor


class TrackNamesEmbeddingsCollector(ICollector):
    def __init__(self, db_engine: AsyncEngine, pool_executor: AioPoolExecutor, openai_client: OpenAIClient):
        self._db_engine = db_engine
        self._openai_client = openai_client
        self._pool_executor = pool_executor

    async def collect(self, missing_tracks: List[MissingTrack]) -> Dict[MissingTrack, Optional[List[float]]]:
        logger.info(f"Starting to collect embeddings for {len(missing_tracks)} tracks")
        results = await self._pool_executor.run(
            iterable=missing_tracks,
            func=self._get_single_name_embeddings,
            expected_type=dict
        )

        return merge_dicts(*results)

    async def _get_single_name_embeddings(self, missing_track: MissingTrack) -> Dict[MissingTrack, Optional[List[float]]]:
        embeddings = await self._openai_client.embeddings.collect(
            text=missing_track.track_name,
            model=EmbeddingsModel.ADA
        )
        return {missing_track: embeddings}
