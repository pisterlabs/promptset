import json

import openai

def get_test_query_data() -> list[tuple[int, str, list[float]]]:
    try:
        with open("data/test_query_data.json", "r") as f:
            test_data = json.loads(f.read())
            print("Loaded test_data from file!")
            return test_data
    except FileNotFoundError:
        test_strings: list[str] = [
        "The sun was setting over the quiet town.",
        "She found joy in the smallest things.",
        "His words echoed through the empty hall.",
        "Every star in the sky tells a story.",
        "The little boy laughed at the clown's antics.",
        "The wind whispered secrets through the trees.",
        "She danced gracefully across the stage.",
        "He spent his days painting vivid landscapes.",
        "Their love story was nothing short of a fairytale.",
        "He stared at the towering mountains in awe.",
        "The smell of fresh coffee filled the kitchen.",
        "Her favorite book was dog-eared and well-loved.",
        "His eyes were as deep as the ocean.",
        "The sound of rain was comforting at night.",
        "She had a knack for solving complex puzzles.",
        "The puppy wagged its tail excitedly.",
        "He was renowned for his excellent piano skills.",
        "She had a dream about flying over the city.",
        "They ventured into the dense, mysterious forest.",
        "She cherished the necklace her grandmother gave her."
        ]
        
        embeddings = openai.Embedding.create(
            input=test_strings,
            model="text-embedding-ada-002",
        )["data"] # type: ignore

        test_data: list[tuple[int, str, list[float]]] = []
        
        for i, embedding in enumerate(embeddings):
            test_data.append(
                (i, test_strings[i], embedding["embedding"])
            )
        
        with open("data/test_query_data.json", "w+") as f:
            f.write(json.dumps(test_data))
        
        print("Downloaded test_data!")
        
        return test_data

if __name__ == "__main__":
    get_test_query_data()