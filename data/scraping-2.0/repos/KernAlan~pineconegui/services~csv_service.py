import json
import os
import tiktoken
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


class CSVService:
    def __init__(self, logger=None):
        load_dotenv()
        self.CSV_FILE = ""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.embeddings_model = "text-embedding-ada-002"
        self.openai_client = OpenAI()
        self.logger = logger
        self.selected_columns = []
        self.max_embedded_tokens = 2000

    def setup_logging(self):
        if self.logger:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_file_path(self, file_path):
        self.CSV_FILE = file_path

    def set_selected_columns(self, columns, main_content_column=None):
        self.selected_columns = columns
        self.main_content_column = (
            main_content_column if main_content_column else columns[0]
        )

    def read_csv_file(self, file_path):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception("CSV file missing or empty")
        return pd.read_csv(file_path)

    def process_csv_dataframe(self, df, main_content_column=None, metadata_column=None):
        # Check if the main content column is in the DataFrame
        if main_content_column not in df.columns:
            self.logger.error(
                f"The column '{main_content_column}' is missing from the CSV file."
            )
            raise Exception(f"CSV file is missing the '{main_content_column}' column.")

        # Check if the metadata column is in the DataFrame
        if metadata_column and metadata_column not in df.columns:
            self.logger.error(
                f"The column '{metadata_column}' is missing from the CSV file."
            )
            raise Exception(f"CSV file is missing the '{metadata_column}' column.")

        # If metadata column is the same as main content column, only select main content column
        if main_content_column != metadata_column:
            df = df[[main_content_column, metadata_column]].copy()
        else:
            df = df[[main_content_column]].copy()

        # Add tokens column to measure the size of the content
        self.logger.info("Calculating tokens and adding to dataframe...")
        df["tokens"] = df[main_content_column].apply(
            lambda x: len(self.ENC.encode(str(x)))
        )

        if df["tokens"].max() > self.max_embedded_tokens:
            raise ValueError(f"Content exceeds {self.max_embedded_tokens} tokens")

        return df

    def create_vectors(self, df, main_content_column=None, metadata_column=None):
        vectors = []
        self.logger.info("Creating vectors...")
        for index, row in df.iterrows():
            content = str(row[main_content_column])
            if content.lower == "nan" or content.lower == "null":
                self.logger.warning(f"Skipping row {index} due to missing content")
                continue

            if len(content) == 0:
                self.logger.warning(f"Skipping row {index} due to empty content")
                continue

            try:
                response = (
                    self.openai_client.embeddings.create(
                        input=content, model=self.embeddings_model
                    )
                    .data[0]
                    .embedding
                )
            except Exception as e:
                self.logger.error(f"Error in embedding content for row {index}: {e}")
                continue

            # Build metadata dictionary from selected columns
            metadata = (
                {metadata_column: str(row[metadata_column])}
                if metadata_column in df.columns
                else {}
            )

            vectors.append({"id": str(index), "values": response, "metadata": metadata})
            self.logger.info(f"Created new JSON object for index {index}")

        return vectors

    def save_json_to_file(self, json_data, file_path):
        try:
            with open(file_path, "w") as json_file:
                json.dump(json_data, json_file)
            self.logger.info(f"JSON file created successfully at {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON file at {file_path}: {e}")

    def main(self):
        self.setup_logging()

        df = self.read_csv_file(self.CSV_FILE)

        df = self.process_csv_dataframe(df)

        vectors = self.create_vectors(df, self.main_content_column)

        json_output = {"vectors": vectors}

        print("Writing JSON to file...")
        self.save_json_to_file(json_output, "output.json")

        backup_path = "backup_output.json"
        if not os.path.exists("output.json"):
            self.logger.warning("Attempting to save JSON to backup location...")
            self.save_json_to_file(json_output, backup_path)
