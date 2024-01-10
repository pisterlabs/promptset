import os
from langchain.document_loaders import UnstructuredPDFLoader
from tqdm import tqdm
import json


def extract_files(data_path="data/PDF", metadata_path="data/metadata.json"):
    pdf_contents = {}
    metadata = json.load(open(metadata_path))
    for file in tqdm(os.listdir(data_path)):
        file_name = file.split(".pdf")[0]
        loader = UnstructuredPDFLoader(data_path + "/" + file)
        data = loader.load()
        text = data[0].page_content
        pdf_contents[file_name] = {"text": text, "doc_nb": metadata["content"][file_name]["doc_nb"], "title": metadata["content"][file_name]["title"], "url": metadata["content"][file_name]["url"]}
    return pdf_contents


if __name__ == "__main__":
    pdf_content = extract_files()
    print(pdf_content)
