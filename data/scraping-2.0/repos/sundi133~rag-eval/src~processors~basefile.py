import os
from typing import List, Callable
from abc import ABC, abstractmethod
from langchain.chains import LLMChain
import random
import time
import json
import pandas as pd


class DataProcessor(ABC):
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.file_extension = os.path.splitext(data_path)[-1].lower()
        self.qa_dict = {}
        self.chunk_size = 2000  # Define the chunk_size attribute here
        self.batch_size = 25
        self.chunk_reference_max_distance = 4

    @abstractmethod
    def parse(self) -> None:
        pass

    @abstractmethod
    def randomize_samples(
        self,
        data: pd.DataFrame,
        sample_size: int,
        products_group_size: int,
        group_columns: List[str],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_qa_pairs(
        self,
        randomized_samples: pd.DataFrame,
        df: pd.DataFrame,
        sample_size: int,
        products_group_size: int,
        group_columns: List[str],
        number_of_questions: int,
        qa_generator: LLMChain,
    ) -> None:
        pass

    @abstractmethod
    def add_output_sample(self, record: dict, **kwargs) -> None:
        pass

    @abstractmethod
    def write(self, file_path: str) -> None:
        pass

    @staticmethod
    def retry_with_exponential_backoff(
        func: Callable,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
    ) -> Callable:
        """Retry a function with exponential backoff."""

        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except (
                    TimeoutError,
                    ConnectionError,
                    ConnectionAbortedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    json.decoder.JSONDecodeError,
                ) as e:
                    num_retries += 1

                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    delay *= exponential_base * (1 + jitter * random.random())
                    time.sleep(delay)
                except Exception as e:
                    raise e

        return wrapper
