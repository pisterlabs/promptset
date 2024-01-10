# In this script we devide a document into chunks, send them to a GPT to see if we can extract possible votings, and then write out the results to json
import os
import json

# Import other needed libraries
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the keys and url from the constants file
import constants

# Set up splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 250)

# Set up an array of files to add to Weaviate
folder_path = "data/helmond/"
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
files = []
files_in_root = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(".pdf")]
for file in files_in_root:
    files.append(file)
for folder in subfolders:
    files_in_subfolder = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".pdf")]
    for file in files_in_subfolder:
        files.append(file)

print(f"Found {len(files)} pdf files in the {folder_path} folder")
for file in files:
    print(file)

if(len(files) == 0):
    print("No pdf files found in the data/ folder. Please add some and try again.")
    exit()

# Make a chunks folder if it doesn't exist
if not os.path.exists(f"{folder_path}/chunks"):
    os.makedirs(f"{folder_path}/chunks")


# Loop over files
for file in files:
    all_chunks = []

    # Process the filename into something we can use as a document name
    file_without_ext = os.path.splitext(file)[0]
    file_without_ext = file_without_ext.split("/")[-1]

    print(f"Processing {file}")
    loader = UnstructuredPDFLoader(file)
    doc = loader.load()
    all_splits = text_splitter.split_documents(doc)
    print(f"Split the document into {len(all_splits)} chunks")

    # Loop over chunks, with index
    for index, split in enumerate(all_splits):
        # Get the page content from the split
        split_content = split.page_content
        # Change newlines to spaces
        split_content = split_content.replace("\n", " ")

        data_object = {
            "document": file,
            "split_index": index,
            "text": split_content
        }

        all_chunks.append(data_object)

    # Write out the chunks to a json file
    with open(f"{folder_path}/chunks/{file_without_ext}.json", 'w') as outfile:
        json.dump(all_chunks, outfile, indent=4, ensure_ascii=False)
        print(f"Written out {len(all_chunks)} chunks to {folder_path}/chunks/{file_without_ext}.json")

