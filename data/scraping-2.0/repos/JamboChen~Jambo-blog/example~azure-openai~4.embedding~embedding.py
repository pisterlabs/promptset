import openai
import pandas

openai.api_key = ""
openai.api_base = ""
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
model: str = "text-embedding-ada-002"


def get_embedding(text: str) -> list:
    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    assert len(embedding) == 1536
    return embedding


def main():
    df = pandas.read_csv("output.csv")
    embeddings = [get_embedding(text) for text in df["content"]]
    df["embedding"] = embeddings
    df.to_csv("docs.csv", index=False)


if __name__ == "__main__":
    import time

    star = time.time()
    main()
    print(f"Time taken: {time.time() - star}")
