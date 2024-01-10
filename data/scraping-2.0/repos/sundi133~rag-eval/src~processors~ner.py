import os
import pandas as pd
import random
import json
import io
import logging
from typing import List
from langchain.chains import LLMChain
from .basefile import DataProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class NERProcessor(DataProcessor):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)
        self.file_extension = os.path.splitext(data_path)[-1].lower()
        self.qa_dict = {}
        self.qa_dict["training_data"] = []
        self.entities_json = {}
        self.qa_array = []
        self.entity_name = ""
        self.topics = []
        self.batch_size = 25

    def set_entity(self, entities_file) -> None:
        with open(entities_file, "r") as json_file:
            self.entities_json = json.load(json_file)
        self.entity_name = self.entities_json["name"]
        self.topics = self.entities_json["keywords"].split(",")
        self.topics = list(map(lambda x: x.strip(), self.topics))

    def parse(self) -> pd.DataFrame:
        key_values = self.entities_json["values"]
        name = self.entities_json["name"]
        modified_df = []
        modified_df.append("sentence")
        out_df = pd.DataFrame(modified_df, columns=["sentence"])
        return out_df

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
        # Initialize a CSV buffer for writing
        for index in range(0, (int)(sample_size / self.batch_size)):
            qa_pair = qa_generator.run(
                sample_size=self.batch_size,
                entity_name=self.entity_name,
            )

            # Log generated questions
            logger.info(
                {
                    "message": "Generated NER training dataset",
                    "data": qa_pair,
                }
            )
            # Split questions by newline and process each question
            question_array = json.loads(qa_pair)
            entity_name = self.entities_json["name"]

            for record in question_array["sentences"]:
                # Log each generated question
                # get index of entity in record
                topic_keyword = random.choice(self.topics)
                record = record["sentence"]
                random_value = random.choice(self.entities_json["values"])
                record = record.replace(
                    f"{entity_name}", f"{topic_keyword} {random_value}"
                )
                entity_index = record.find(f"{random_value}")
                entity_length = len(f"{random_value}")

                data = {
                    "text": record,
                    "entities": [
                        {
                            "start": entity_index,
                            "end": entity_index + entity_length,
                            "label": entity_name,
                            "value": random_value,
                        }
                    ],
                }

                logger.info(
                    {
                        "message": "Generated ner training dataset",
                        "question_answer": data,
                    }
                )
                self.add_output_sample(data)

        return self.qa_dict

    def add_output_sample(self, record: json) -> None:
        self.qa_array.append(record)

    def write(self, file_path: str) -> None:
        with open(file_path, "w") as output_file:
            json.dump(self.qa_array, output_file, indent=4)
