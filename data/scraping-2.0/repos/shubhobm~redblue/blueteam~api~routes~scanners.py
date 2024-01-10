from dotenv import load_dotenv
from fastapi import APIRouter
from joblib import load
import logging
import openai
import os
import warnings
import time

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

HOME = os.path.join(os.path.dirname(__file__), "../../..")
load_dotenv(f"{HOME}/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "embeddings_openai"  # Collection name
OPENAI_ENGINE = "text-embedding-ada-002"  # Which engine to use
all_classes = ["security", "toxicity", "stereotype"]


# Search the database based on input text
def embed(text):
    return openai.Embedding.create(input=text, engine=OPENAI_ENGINE)["data"][0][
        "embedding"
    ]


class Scanner:
    def __init__(self):
        self.models = {}
        for vuln_class in all_classes:
            self.models[vuln_class] = load(
                f"{HOME}/blueteam/api/models/clf_{vuln_class}.joblib"
            )

    def scan(self, prompt):
        start = time.time()
        embedding = embed(prompt)
        t1 = time.time() - start

        start = time.time()
        output = {"prompt": prompt, "detection": []}
        for vuln_class in all_classes:
            output["detection"].append(
                {
                    "class": vuln_class,
                    "flag": str(self.models[vuln_class].predict([embedding])[0]),
                    "score": self.models[vuln_class].predict_proba([embedding])[0][1],
                }
            )
        t2 = time.time() - start
        return {
            "output": output,
            "latency (ms)": {
                "embedding": round(t1 * 1000, 1),
                "classification": round(t2 * 1000, 1),
            },
        }


scan_router = APIRouter()
scanner = Scanner()


# Start a new garak evaluation
@scan_router.post("/scan")
async def scan(prompt: str):
    logging.info(f"Received a new scan request with prompt: {prompt}")
    output = scanner.scan(prompt)
    logging.info(f"Scan results: {output}, of type {type(output)}")
    return output
