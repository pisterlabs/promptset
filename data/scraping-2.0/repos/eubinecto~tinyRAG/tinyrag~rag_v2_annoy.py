from pathlib import Path
import yaml
import openai
from annoy import AnnoyIndex


class RAGVer2Annoy:
    def __init__(self) -> None:
        with open(Path(__file__).resolve().parent / "openai27052023.yaml", 'r') as f:
            self.openai_paper = yaml.safe_load(f)
        self.sentences = self.openai_paper['sentences']
        # from texts to embeddings (numeric representations of the texts)
        embeddings = [
            r['embedding']
            for r in openai.Embedding.create(input=self.sentences, model='text-embedding-ada-002')['data']
        ] 
        hidden_size = len(embeddings[0])
        self.index = AnnoyIndex(hidden_size, 'angular')  #  "angular" =  cosine
        for i, e in enumerate(embeddings): 
            self.index.add_item(i , e)
        self.index.build(10)  # build 10 trees for efficient search
                # second - build an annoy index

    def __call__(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        # get an embedding for the query givens
        embedding =  openai.Embedding.create(input = [query], model='text-embedding-ada-002')['data'][0]['embedding']
        # get nearest neighbors by vectors
        indices, distances = self.index.get_nns_by_vector(embedding, n=k, include_distances=True)
        results =  [ 
            (self.sentences[i], d)
            for i, d in zip(indices, distances)
        ]
        return results

    