import logging
import sys
import os

from dotenv import find_dotenv, load_dotenv
from inquirer import answer_query
from langchain.embeddings.openai import OpenAIEmbeddings
from helper import get_dbs
from api import RESPONSE_TYPE_GENERAL, RESPONSE_TYPE_DEPTH

# Add the relative path of the directory where preprocessor.py is located
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../backend/src"))

# Now you should be able to import create_embeddings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

load_dotenv(find_dotenv())


def main():
    query_memory = []
    db_general, db_in_depth, voting_roll_df = get_dbs()

    while True:
        query_input = input("Enter your query with response type (or 'quit' to exit): ")
        if query_input.lower() == "quit":
            break

        # Split the query_input into the response_type and the query
        query_parts = query_input.split(": ", 1)
        response_type = query_parts[0]
        query = query_parts[1]

        # map response type to the required format
        response_type_map = {
            "General Summary": RESPONSE_TYPE_GENERAL,
            "In-Depth Response": RESPONSE_TYPE_DEPTH,
        }
        if response_type not in response_type_map:
            print(
                "Invalid response type. Please start your query with either 'General Summary' or 'In-Depth Response'."
            )
            continue

        response = answer_query(
            query,
            response_type_map[response_type],
            voting_roll_df,
            db_general,
            db_in_depth,
        )
        print(response)
        query_memory.append(query)

    print("Query memory:")
    print(query_memory)


if __name__ == "__main__":
    main()
