import json
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List

import openai
from openai.error import RateLimitError

from privacy_fingerprint.common import Record
from privacy_fingerprint.common.config import (
    load_experiment_config,
    load_global_config,
)
from privacy_fingerprint.common.db import get_session
from privacy_fingerprint.common.models import LLM, generate_id


class LMGenerator:
    """A base language model generator"""

    def __init__(self):
        config = load_experiment_config()
        self.model = config.openai.model
        self.temperature = config.openai.temperature
        self.max_tokens = config.openai.max_tokens
        self.prompt = config.openai.prompt

    def make_text(self, record: Dict[str, Any]) -> str:
        return list(
            self.generate_text(
                [
                    record,
                ]
            )
        )[0]

    def generate_text(self, records: List[Dict[str, Any]]) -> Iterable[str]:
        """Convert a series of structured records into unstructured clinical notes

        :param records: List of structured records
        :returns: Iterable of unstructured clinical notes
        """
        config = load_global_config()
        cache_miss_records = self.get_cache_miss_records(records)
        batch = []
        for record in cache_miss_records:
            batch.append(record)
            if len(batch) < config.openai.batch_size:
                continue
            self.batch_query_api(batch)
            batch = []
        if len(batch) > 0:
            self.batch_query_api(batch)
        return self.records_from_cache(records)

    def batch_query_api(self, batch: List[Record]) -> Iterable[str]:
        """Send a batch of records to the API for conversion to unstructured text

        :param batch: List of structured records
        :returns: Iterable of unstructured clinical notes"""
        config = load_global_config()
        query = list(self.prepare_query(batch))
        delay = config.openai.delay_on_error
        attempts = 0
        while attempts < config.openai.retry_attempts:
            try:
                api_results = list(self._call_api(query))
                with get_session()() as session:
                    for record, result in zip(batch, api_results):
                        session.add(
                            LLM(
                                id=self.record_id(record),
                                prompt=self.prompt,
                                encounter=record,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                model=self.model,
                                llm_response=result,
                                date_time=datetime.now(),
                            )
                        )
                    session.commit()
                return api_results
            except RateLimitError:
                attempts += 1
                time.sleep(delay)
                delay *= config.openai.backoff_on_error
                if delay > config.openai.max_delay_on_error:
                    delay = config.openai.max_delay_on_error

    def prepare_query(self, batch: Iterable[Dict[str, Any]]) -> Iterable[str]:
        """Convert records to a set of queries

        :param batch: List of structured records
        :returns: Iterable of queries to send to the openai API"""
        for record in batch:
            patient_record = json.dumps(record, indent=4)
            query = f"{self.prompt}\n{patient_record}"
            yield query

    def _call_api(self, query: List[str]) -> Iterable[str]:
        """Call the openai API

        :param query: List of queries to send to the openai API
        :returns: Iterable of text results from the openai API"""
        config = load_global_config()
        openai.api_key = config.openai.api_key
        openai_result = openai.Completion.create(
            model=self.model,
            prompt=query,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        for result in openai_result.choices:
            yield result.text

    def get_cache_miss_records(self, records):
        """Get records that have not yet been saved to the cache

        :param records: List of records
        :returns: List of records not in the cache"""
        record_lookup = {self.record_id(record): record for record in records}
        with get_session()() as session:
            result = (
                session.query(LLM.id)
                .where(LLM.id.in_(record_lookup.keys()))
                .all()
            )
        return [
            record_lookup[record_id]
            for record_id in set(record_lookup.keys())
            - set([i[0] for i in result])  # noqa: W503
        ]

    def records_from_cache(self, records):
        """Get records from the cache

        :param records: List of records
        :returns List of unstructured medical notes"""
        record_ids = [self.record_id(record) for record in records]
        Session = get_session()
        with Session() as session:
            result = session.query(LLM).where(LLM.id.in_(record_ids)).all()
        result_lookup = {r.id: r.llm_response for r in result}
        for r_id in record_ids:
            yield result_lookup[r_id]

    def record_id(self, record: Record) -> str:
        """Generate a record ID

        :param record: A record
        :returns: Hashed ID for API query"""
        return generate_id(
            prompt=self.prompt,
            encounter=record,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model=self.model,
        )
