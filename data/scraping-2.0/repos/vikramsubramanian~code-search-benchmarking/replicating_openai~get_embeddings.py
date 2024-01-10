
import json
from openai import AzureOpenAI
import time
import numpy as np




class OpenAIEmbeddings:
    def __init__(self):
        """
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        """

        self._client = AzureOpenAI(
            api_version=OPEN_AI_API_VERSION,
            azure_endpoint=OPEN_AI_BASE,
            api_key=OPENAI_API_KEY
        )

    def get_embedding(self, chunk: str):
        return (
            self._client.embeddings.create(
                input=chunk, model=EMBEDDING_MODEL
            )
            .data[0]
            .embedding
        )

    def get_embeddings_save_to_file(self, code, doc, output_file):

            embeddings = []
            for i, (code_chunk, doc_chunk) in enumerate(zip(code, doc)):
                print(f"Processing chunk {i}")
                try:
                    code_embedding = self.get_embedding(code_chunk)
                    doc_embedding = self.get_embedding(doc_chunk)
                except Exception as e:
                    print(e)
                    time.sleep(60)
                    continue
                embeddings.append(
                    {
                        "code": code_chunk,
                        "docstring": doc_chunk,
                        "code_embedding": code_embedding,
                        "doc_embedding": doc_embedding,
                    }
                )
            with open(output_file, "w") as f:
                json.dump(embeddings, f)



def load_embeddings_from_file(path_to_embeddings):
    with open(path_to_embeddings, "r") as f:
        return json.load(f)


def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_mrr(embeddings):
    ranks = []
    for item in embeddings:
        similarities = [
            np.dot(item["doc_embedding"], other["code_embedding"])
            / (
                np.linalg.norm(item["doc_embedding"])
                * np.linalg.norm(other["code_embedding"])
            )
            for other in embeddings
        ]
        sorted_similarities = sorted(similarities, reverse=True)

        correct_rank = (
            sorted_similarities.index(
                np.dot(item["doc_embedding"], item["code_embedding"])
                / (
                    np.linalg.norm(item["doc_embedding"])
                    * np.linalg.norm(item["code_embedding"])
                )
            )
            + 1
        )

        ranks.append(correct_rank)

    MRR = np.mean([1 / rank for rank in ranks])
    return MRR



if __name__ == "__main__":
    # embeddings = OpenAIEmbeddings()
    # path_to_codesearchnet = "/Users/vikram/Downloads/python/python/final/jsonl/valid/python_valid_0.jsonl"
    # with open(path_to_codesearchnet, "r") as f:
    #     lines = f.readlines()
    #     code_chunks = [json.loads(line)["code"] for line in lines]
    #     doc_chunks = [json.loads(line)["docstring"] for line in lines]


    #     embeddings.get_embeddings_save_to_file(code_chunks, doc_chunks, "python_embs.json")


    print(compute_mrr(load_embeddings_from_file("python_embs.json")))
