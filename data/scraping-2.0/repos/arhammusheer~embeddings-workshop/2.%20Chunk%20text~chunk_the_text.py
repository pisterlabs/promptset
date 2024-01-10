import openai
import tiktoken
import pinecone
import os

# Credentials
OPENAI_API_KEY = "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"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

PINECONE_API_KEY = "d22a004e-e747-4ffb-b5b5-b39da101eb2c"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX = "example-index"

# Set up Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

if PINECONE_INDEX not in pinecone.list_indexes():
    # Create the index
    pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine")

# Get the index
index = pinecone.Index(PINECONE_INDEX)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


def text_splitter(content, token_count=256, overlap=30):
    chunks = []
    # Tiktoken
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Split the text into tokens
    tokens = encoding.encode(content)

    # Get the total number of tokens
    token_length = len(tokens)
    if token_count == 0:
        return [content]

    # Get the number of chunks
    chunk_count = round(token_length / token_count)
    if chunk_count == 0:
        return [content]

    # Get the number of tokens per chunk
    tokens_per_chunk = round(token_length / chunk_count)

    # Get the number of overlapping tokens
    overlap_tokens = round(tokens_per_chunk / overlap)

    # Get the start and end token for each chunk
    for chunk in range(chunk_count):
        start = chunk * tokens_per_chunk
        end = start + tokens_per_chunk + overlap_tokens
        chunks.append(encoding.decode(tokens[start:end]))

    return chunks


def recursive_file_read(directory, contents=[]):
    # Get the files in the directory
    files = os.listdir(directory)

    # Loop through the files
    for file in files:
        # Get the file path
        file_path = os.path.join(directory, file)

        # Check if the file is a directory
        if os.path.isdir(file_path):
            # Read the directory
            recursive_file_read(file_path, contents)
        else:
            # Read the file if .md
            if ".md" not in file_path:
                continue
            file = open(file_path, "r")
            content = file.read()
            file.close()
            contents.append(
                {"text": content, "page": file_path.split("/")[-1].replace(".md", "")}
            )

    return contents


def vectorize(text):
    # Vectorize text
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDINGS_MODEL,
    )
    return response.data[0].embedding


def upload_to_pinecone(items):
    # Upload to pinecone
    index.upsert(items)


# Main
if __name__ == "__main__":
    file_path = "content"
    content = recursive_file_read(file_path)
    print(len(content))

    # Save content to content/
    for page in content:
        # Create the file
        file = open(f"./content/page_{page['page']}.txt", "w")
        # Write the page to the file
        file.write(page["text"])
        # Close the file
        file.close()

    # Split the text into chunks
    chunks = []
    for page in content:
        # Chunk the page
        page_chunks = text_splitter(page["text"])
        for chunk in page_chunks:
            chunks.append({"text": chunk, "page": page["page"]})

    # Save chunks to files/
    for i, chunk in enumerate(chunks):
        # Create the file
        file = open(f"./chunks/chunk_{i}_{chunk['page']}.txt", "w")
        # Write the chunk to the file
        file.write(chunk["text"])
        # Close the file
        file.close()
    print(f"Chunk {i} saved!")

    BATCH_SIZE = 100

    completed = 0

    # Vectorize chunks
    for i in range(0, len(chunks), BATCH_SIZE):
        # Pinecone format (ID, vector, metadata)
        batch = [
            (
                f"{i}_{chunk['page']}",
                vectorize(chunk["text"]),
                {"page": chunk["page"], "text": chunk["text"]},
            )
            for i, chunk in enumerate(chunks[i : i + BATCH_SIZE])
        ]

        # Upload to pinecone
        upload_to_pinecone(batch)

        # Print progress
        completed += BATCH_SIZE
        print(f"{completed}/{len(chunks)} chunks completed!")
