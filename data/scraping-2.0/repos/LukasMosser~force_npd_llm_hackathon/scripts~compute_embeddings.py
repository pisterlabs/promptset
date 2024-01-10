from openai import AzureOpenAI
from openai.types import Embedding
from dotenv import load_dotenv
import os
from datasets import load_dataset, ClassLabel, Value, Features, Sequence
from json.decoder import JSONDecodeError
from openai import BadRequestError
import json
from tqdm.auto import tqdm 
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken
import time

CORPORA = {
    "NL_NOG_PR": "Netherlands - Netherlands Oil & Gas Portal reports.csv",
    "NO_NPD_DI": "Norway - Diskos reports.csv",
    "NO_NPD_RR": "Norway - Norwegian Petroleum Directorate relinquishment reports.csv",
    "UK_NTA_NDR": "UK - North Sea Transition Authority NDR reports.csv",
    "UK_NTA_RR": "UK - North Sea Transition Authority relinquishment reports.csv"
}

features = Features({'doc_id': Value('string'), 
                     'meta': Features({'_id': Value(dtype='string', id=None), 
                                               'corpus': Value(dtype='string', id=None),
                                                 'langdetect': Value(dtype='string', id=None), 
                                                 'possible_lanaguage': Value(dtype='string', id=None), 
                                                 'content_could_be_natural_language': Value(dtype='string', id=None)
                                                 }), 
                    'raw_content': Features({'Document Corpus': Value(dtype='string', id=None),
                                                 'Filename': Value(dtype='string', id=None),
                                                 'Content': Value('string')
                                                 })
})

force_llm_dataset_scrubbed = load_dataset("json", data_files="./data/force_llm_corpus_scrubbed_embedding_docs.jsonl", features=features)

enc = tiktoken.get_encoding("cl100k_base")

def fetch_dataset_records(dataset, batch_size=16):
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        yield dataset[start_idx:end_idx]

# Load the shared environment variables, not secrets
load_dotenv(".env.shared")
load_dotenv(".env.secret")

openai_client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

def make_embedding_text(batch_content):
    corpus = CORPORA[batch_content["Document Corpus"]]
    content = batch_content["Content"]
    filename = batch_content["Filename"]
    embedding_text = f"FILENAME: {filename}| CORPUS: {corpus} | TEXT: {content}"
    return embedding_text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(100))
def embeddings_with_backoff(**kwargs):
    return openai_client.embeddings.create(**kwargs)

total_tokens = 0
embeddings_list = []
batch_size = 16
start_time = time.time()
with open("./data/force_llm_corpus_scrubbed_embeddings.jsonl_", "w+") as f:
    prg_bar = tqdm(fetch_dataset_records(force_llm_dataset_scrubbed['train'], batch_size=batch_size), total=len(force_llm_dataset_scrubbed['train']) // batch_size)
    for batch in prg_bar:
        try:
            batch_content = []
            for x in batch["raw_content"]:
                batch_content.append(x)
            
            embedding_text = [make_embedding_text(x) for x in batch_content]
            total_tokens += sum([len(enc.encode(x)) for x in embedding_text])
            embeddings = embeddings_with_backoff(input=embedding_text, model=os.environ["ADA002_DEPLOYMENT"])
            current_time = time.time()
            time.sleep(0.5)
            avg_tokens_per_second = total_tokens / (current_time - start_time)
            prg_bar.set_description_str(f"Total tokens: {total_tokens}. Average tokens per second: {avg_tokens_per_second}. Max tokens per second: {350_000/60.}.")
            for doc_id, embedding in zip(batch["doc_id"], embeddings.data):
                f.write(json.dumps({"doc_id": doc_id, "embedding": embedding.embedding}) + "\n")

        except (JSONDecodeError, BadRequestError):
            try: 
                for i, id, text in enumerate(zip(batch["doc_id"], batch["raw_content"])):
                    try:
                        total_tokens += sum([len(enc.encode(x)) for x in embedding_text])
                        embeddings = embeddings_with_backoff(input=text, model=os.environ["ADA002_DEPLOYMENT"])
                        f.write(json.dumps({"doc_id": id, "embedding": embedding.data[0].embedding}) + "\n")
                    except (JSONDecodeError, BadRequestError, ValueError):
                        f.write(json.dumps({"doc_id": id, "embedding": []}) + "\n")
            except ValueError:
                print("Had to skip: ", batch["doc_id"])
