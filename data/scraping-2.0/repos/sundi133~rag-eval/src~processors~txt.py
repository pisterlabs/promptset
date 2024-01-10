import os
import pandas as pd
import json
import io
import logging
import random
import time

from langchain.chains import LLMChain
from typing import List
from .basefile import DataProcessor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class TXTProcessor(DataProcessor):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)
        self.file_extension = os.path.splitext(data_path)[-1].lower()
        self.qa_dict = {}
        self.qa_array = []
        self.batch_size = 25
        self.chunk_size = 2000

    def parse(self) -> pd.DataFrame:
        with open(self.data_path, "r") as f:
            content = f.read()
        chunks = [
            content[x : x + self.chunk_size]
            for x in range(0, len(content), self.chunk_size)
        ]
        data = [x.strip() for x in chunks]

        df = pd.DataFrame({"chunk": data})
        df["chunk"] = df["chunk"].apply(lambda x: x.strip())
        df = df[df["chunk"].notna() & (df["chunk"] != "")].reset_index(drop=True)

        logger.info(
            {
                "message": "Parsed data",
                "data": df,
            }
        )
        return df

    def randomize_samples(
        self,
        data: pd.DataFrame,
        sample_size: int,
        products_group_size: int,
        group_columns: List[str],
    ) -> pd.DataFrame:
        if sample_size > data.shape[0]:
            sample_size = data.shape[0]
        return data.sample(n=sample_size, random_state=42)

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
        for _index, group_row in randomized_samples.iterrows():
            try:
                records = group_row["chunk"]

                if len(records) < 100:
                    continue

                if (
                    number_of_questions > self.batch_size
                ):  # too many questions might cause token limit error
                    number_of_questions = self.batch_size

                logger.info(
                    {
                        "message": "check qa_generator",
                        "qa_generator": qa_generator.prompt.input_variables,
                    }
                )
                if (
                    "chunk_reference_first" in qa_generator.prompt.input_variables
                    and "chunk_reference_second" in qa_generator.prompt.input_variables
                ):
                    # Define window boundaries based on current index
                    window_indices = [
                        _index + i
                        for i in range(
                            -self.chunk_reference_max_distance,
                            self.chunk_reference_max_distance,
                        )
                        if 0 <= _index + i < randomized_samples.shape[0] and i != 0
                    ]
                    desired_index = window_indices[-1]
                    row_content = randomized_samples.iloc[desired_index]

                    # Check if "chunk" column exists, otherwise access the entire row
                    chunk_reference_second = row_content["chunk"]

                    qa_pair = qa_generator.run(
                        chunk_reference_first=records,
                        chunk_reference_second=chunk_reference_second,
                        number_of_questions=number_of_questions,
                    )
                    records = (
                        records
                        + "\n\n"
                        + "Distant reference chunk: "
                        + chunk_reference_second
                    )
                else:
                    qa_pair = qa_generator.run(
                        products=records,
                        number_of_questions=number_of_questions,
                    )

                question_array = json.loads(qa_pair)
                logger.info(
                    {
                        "message": "Generated question & answer pair length",
                        "questions": len(question_array),
                    }
                )
                for record in question_array:
                    logger.info(
                        {
                            "message": "Generated question",
                            "question_answer": record,
                            "reference": records,
                        }
                    )
                    self.add_output_sample(record, chunk=records)

            except Exception as e:
                logger.error(
                    {
                        "message": "Error generating question",
                        "error": str(e),
                    }
                )
                continue
        return self.qa_dict

    def add_output_sample(self, record: json, chunk: str) -> None:
        self.qa_array.append(
            {
                "question_answer": record,
                "reference": chunk,
            }
        )

    def write(self, file_path: str) -> None:
        logger.info(
            {
                "message": "Writing generated questions to file",
                "file_path": file_path,
            }
        )
        with open(file_path, "w") as output_file:
            json.dump(self.qa_array, output_file, indent=4)
