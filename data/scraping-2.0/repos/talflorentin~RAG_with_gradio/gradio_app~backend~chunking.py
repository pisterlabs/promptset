import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

FILES_DUMP_FOLDER = 'split_files_dump'
directory_path = 'docs_dump'
directory_full_path = os.path.join(os.getcwd(), directory_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=200, chunk_overlap=20, disallowed_special=()
# )

files_splits = {}

for filename in os.listdir(directory_full_path):
    if filename.endswith('.txt'):
        # Construct the full path of the text file
        file_path = os.path.join(directory_full_path, filename)

        with open(file_path, 'r') as file:
            file_contents = file.read()
            # print(f"Contents of {filename}:\n{file_contents}\n")

            texts = text_splitter.create_documents([file_contents])

            files_splits[filename[:-4]] = texts


output_path = FILES_DUMP_FOLDER
output_full_path = os.path.join(os.getcwd(), output_path)
os.makedirs(output_full_path, exist_ok=True)

for key, string_list in files_splits.items():
    for index, doc in enumerate(string_list):
        file_path = os.path.join(output_full_path, f"{key}_{index}.txt")
        with open(file_path, 'w') as file:
            file.write(doc.page_content)

print('Finished')
