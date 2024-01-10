from typing import List, Optional
from logger import logger
import csv
import os
import openai

sample_file = "../Sample/sample.txt"
csv_file = "../Database/ada002.csv"

def get_file_content(filename) -> List[str]:
    with open(file=filename, mode="r") as f:
        content = f.read()
    lines = [line for line in content.split("\n") if line != ""]
    logger.info("Read %d lines from %s", len(lines), filename)
    return lines

def get_key() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info("OpenAI API Key: %s", openai.api_key)

def create_vector(lines: List[str], model: str ="text-embedding-ada-002") -> Optional[List[List[float]]]:
    try:
        response = openai.Embedding.create(
            model=model,
            input=lines
        )
        if response and 'data' in response and len(response['data']):
            vectors = [data['embedding'] for data in response['data']]
            logger.info("Embedding created successfully")
            logger.info("Generated %d vectors", len(vectors))
            return vectors
        else:
            logger.warning("Error: No data returned from the API")
            raise ValueError("No data returned")

    except Exception as e:
        logger.warning("Error: %s", e)
        return None

def save_vector(vectors: List[List[float]]) -> None:
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        for vector in vectors:
            writer.writerow(vector)
    logger.info("Vectors have been written to %s" % csv_file)

if __name__ == "__main__":
    get_key()
    lines = get_file_content(sample_file)
    vectors = create_vector(lines)
    save_vector(vectors)
