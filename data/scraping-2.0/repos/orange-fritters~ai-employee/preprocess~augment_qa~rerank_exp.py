import json
import openai
openai.api_key_path = "config.txt"


def get_embed(texts):
    result = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=texts)
    return result


if __name__ == "__main__":
    texts = ["I like to eat pizza", "pizza"]
    result = get_embed(texts)
    with open("embedding.json", "w") as f:
        json.dump(result, f, indent=4)
