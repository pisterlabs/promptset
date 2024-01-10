import openai
import csv
import os 

# Initialize the OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

def compute_embeddings(texts):
    response =openai.Embedding.create(
        model="text-embedding-ada-002",
         input=texts
    )
    return [d["embedding"] for d in response.data]

def read_file_in_batches(filename, batch_size):
    with open(filename, 'r') as file:
        batch = []
        for line in file:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    filename = "/Users/tsg/Downloads/wikisent2.txt"
    batch_size = 100  # Adjust as needed
    i = 0
    skip = 8961

    with open("sentences.2.csv", 'w') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["sentence", "embedding"])

        for batch in read_file_in_batches(filename, batch_size):
            if i < skip:
                i += 1
                continue
            embeddings = compute_embeddings(batch)
            for sentence, embedding in zip(batch, embeddings):
                writer.writerow([sentence, embedding])
            i += 1
            print("Wrote ", i*batch_size, " records")


if __name__ == "__main__":
    main()