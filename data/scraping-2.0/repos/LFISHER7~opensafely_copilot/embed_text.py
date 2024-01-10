import argparse
import os
import re
import time

import openai
import pinecone
from dotenv import load_dotenv


def get_docs_link(file_path, input_dir="copilot/data/doc-sections"):
    """
    Converts a file path to a documentation link.

    Args:
        file_path (str): Path to the input file.
        input_dir (str): Path to the input directory.
    Returns:
        str: A documentation link.
    """
    # Convert file path to a URL path
    
    converted_path = file_path.replace(f"{input_dir}\\", "")

    converted_path = re.sub(r"_section_", "/", converted_path)
    converted_path = converted_path.replace(".txt", "")
    converted_path = converted_path.replace("_", "-")
    converted_path = converted_path.lower()

    # Add anchor tag to URL
    converted_path = re.sub(r"(.*)/(.*)", r"\1/#\2", converted_path)

    # Remove "#no-header" from URL
    converted_path = re.sub(r"(.*)/#no-header", r"\1", converted_path)
    
    # Return URL
    return f"https://docs.opensafely.org/{converted_path}"

def process_files(input_dir, index_name):
    """
    Processes input files and generates embeddings.

    Args:
        input_dir (str): Path to the input directory.
        index_name (str): Name of the Pinecone index.
    """
    # Get list of input files
    files = os.listdir(input_dir)
    txt_files = [file for file in files if file.endswith(".txt")]

    # Loop through input files and generate embeddings
    for txt_file in txt_files:
        input_file_path = os.path.join(input_dir, txt_file)
        with open(input_file_path, "r", encoding="utf-8") as file:
            # Read input file
            plain_text = file.read()

            # Get documentation link
            link = get_docs_link(input_file_path)
           
           
            print(f"Processing {input_file_path}...")
            # Generate embedding
            gpt3_embedding = get_embedding(plain_text)

            # Upload embedding to Pinecone
            if gpt3_embedding is not None:
                key = link
                index = pinecone.Index(index_name)
                index.upsert([(key, gpt3_embedding,  {"text": plain_text})])
                # Wait for 1 second to avoid rate limit
                time.sleep(1)




def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding for a given text using OpenAI API.

    Args:
        text (str): The input text.
        model (str): The name of the OpenAI model to use for generating embeddings.


    Returns:
        list: A list of 768 floating point numbers representing the embedding.
    """
    # Remove newline characters from text
    text = text.replace("\n", " ")

    # Check if text is empty
    if text == "":
        return None

    # Generate embedding using OpenAI API
    embedding = openai.Embedding.create(input=[text], model=model)
    return embedding["data"][0]["embedding"]

def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: A namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="copilot/data/doc-sections",
        help="Path to the input directory.",
    )
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    # Parse command line arguments
    args = parse_args()

    # Set input and output directories
    input_dir = args.input_dir

    load_dotenv()

    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone and create an index
    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536)

    # Initialize OpenAI API
    openai.api_key = openai_api_key

    # Process input files
    process_files(input_dir=input_dir, index_name=index_name)
if __name__ == "__main__":
    main()