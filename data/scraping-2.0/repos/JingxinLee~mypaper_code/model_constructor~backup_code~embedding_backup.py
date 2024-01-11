import os
from langchain.document_loaders import UnstructuredMarkdownLoader
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Function to process a batch of Markdown files
def process_markdown_batch(markdown_files):
    batch_docs = []
    for markdown_file_path in markdown_files:
        markdown_loader = UnstructuredMarkdownLoader(markdown_file_path)
        batch_docs.extend(markdown_loader.load())
    return batch_docs



# Get the list of Markdown files to process
markdown_files_to_process = []
# Specify the root directory where you want to search for PDF files
root_directory = "model_constructor/MarkdownFiles"

for root, dirs, files in os.walk(root_directory):
    markdown_files_to_process.extend(
        [os.path.join(root, file) for file in files if file.lower().endswith(".md")]
    )

print("markdown_files_to_process: ", markdown_files_to_process)


# Set the batch size (number of files to process in each batch)
batch_size = 1
# Initialize an empty list to store loaded documents
docs = []
total_files = len(markdown_files_to_process)
processed_files = 0

docs_embeddings = {}
# Iterate through the Markdown files in batches
for i in range(0, total_files, batch_size):
    batch = markdown_files_to_process[i : i + batch_size]
    batch_docs = list(map(process_markdown_batch, [batch]))
    for batch_result in batch_docs:
        docs.extend(batch_result)
        print(docs)
        processed_files += len(batch)
        print(f"Processed {processed_files} / {total_files} files")

# Function to create embeddings
def create_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


for doc in docs:
    docs_embeddings[doc.metadata["source"]] = create_embedding(doc.page_content)

print(docs_embeddings)
