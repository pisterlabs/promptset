from openai import OpenAI
import torch

client = OpenAI()

# consume proposed constitution and output normalized scores with existing constitutions
def get_normalized_scores(proposed_constitution, constitutions):
    proposed_constitution_embedding = client.embeddings.create(
        input=proposed_constitution,
        model="text-embedding-ada-002"
    )

    proposed_constitution_tensor = torch.tensor(proposed_constitution_embedding.data[0].embedding)

    cosine_similarities = []
    for constitution in constitutions:
        constitution_embedding = client.embeddings.create(
            input=constitution,
            model="text-embedding-ada-002"
        )
        constitution_tensor = torch.tensor(constitution_embedding.data[0].embedding)
        cosine_similarity = torch.nn.functional.cosine_similarity(proposed_constitution_tensor, constitution_tensor, dim=0)
        cosine_similarities.append(cosine_similarity)

    cosine_similarities = torch.tensor(cosine_similarities)
    normalized_cosine_similarities = (cosine_similarities - torch.mean(cosine_similarities)) / torch.std(cosine_similarities)

    normalized_scores = {i: score.item() for i, score in zip(range(len(constitutions)), normalized_cosine_similarities)}
    return normalized_scores
