import logging
import concurrent
from datasets import load_dataset
from dotenv import load_dotenv
import openai
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Application started")

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "Set your OPENAI_API_KEY via export OPENAI_API_KEY=... or in a .env file"
    )
openai.api_key = os.environ["OPENAI_API_KEY"]
logging.info("OpenAI API key loaded")

EMBEDDINGS_MODEL = "text-embedding-ada-002"


def extract_texts_from_folder(folder_path):
    logging.info(f"Extracting texts from folder: {folder_path}")
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            with open(os.path.join(folder_path, filename), "r") as file:
                texts[filename] = file.read()
    return texts


def get_embedding_for_text(filename, text):
    logging.info(f"Getting embedding for {filename}")
    response = openai.Embedding.create(input=text, model=EMBEDDINGS_MODEL)
    embedding = response["data"][0]["embedding"]
    return {filename: {"embedding": embedding, "text": text}}


def get_embeddings(texts):
    embeddings_data = {}
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(get_embedding_for_text, filename, text): filename
            for filename, text in texts.items()
        }
        for future in concurrent.futures.as_completed(future_to_file):
            embeddings_data.update(future.result())
    return embeddings_data


def save_embeddings_to_file(embeddings, file_path):
    logging.info(f"Saving embeddings to file: {file_path}")
    pickle.dump(embeddings, open(file_path, "wb"))


def process_my_solutions():
    folder_path = "solutions"
    texts = extract_texts_from_folder(folder_path)
    embeddings = get_embeddings(texts)
    save_embeddings_to_file(embeddings, "data/embeddings.pkl")


def process_huggingface_solutions():
    dataset = load_dataset("mhhmm/leetcode-solutions-python")
    df = dataset["train"].to_pandas()
    texts = df["code_with_problem"].to_dict()
    embeddings_data = get_embeddings(texts)
    save_embeddings_to_file(embeddings_data, "data/hf_embeddings.pkl")


process_my_solutions()
process_huggingface_solutions()

logging.info("Application completed successfully")
