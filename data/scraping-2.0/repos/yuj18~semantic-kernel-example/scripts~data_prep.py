# This is a convenient script to parse the PDF files, chunk them into sections,
# and index them into Azure Cognitive Search index. This script is based on:
# https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/scripts/prepdocs.py

import argparse
import base64
import os
import re

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)
from dotenv import dotenv_values
from pypdf import PdfReader, PdfWriter

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100


def get_document_text(filename, output_dir=None):
    offset = 0
    page_map = []
    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
        if output_dir:
            writer = PdfWriter()
            writer.add_page(p)
            output_pdf_path = os.path.join(
                output_dir,
                f"{os.path.basename(filename).split('.')[0]}_{page_num}.pdf",
            )
            writer.write(output_pdf_path)

    return page_map


def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    def find_page(offset):
        num_pages = len(page_map)
        for i in range(num_pages - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return num_pages - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while (
                end < length
                and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT
                and all_text[end] not in SENTENCE_ENDINGS
            ):
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word  # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while (
            start > 0
            and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT
            and all_text[start] not in SENTENCE_ENDINGS
        ):
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (
            last_table_start > 2 * SENTENCE_SEARCH_LIMIT
            and last_table_start > section_text.rfind("</table")
        ):
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))


def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode("utf-8")).decode("ascii")
    return f"file-{filename_ascii}-{filename_hash}"


def create_embedding(text, openai_embedding_model):
    try:
        embedded_text = openai.Embedding.create(
            input=text, deployment_id=openai_embedding_model
        )["data"][0]["embedding"]
    except Exception as e:
        print(f"Error creating embedding for text: {text} with error: {e}")
        embedded_text = None
    return embedded_text


def create_sections(filename, page_map, output_dir, openai_embedding_model=None):
    file_id = filename_to_id(filename)
    for i, (content, page_num) in enumerate(split_text(page_map)):
        section = {
            "id": f"{file_id}-page-{i}",
            "content": content,
            "content_vector": create_embedding(content, openai_embedding_model),
            "category": args.category,
            "source_page": os.path.join(
                output_dir, f"{filename.split('.')[0]}_{str(page_num)}.pdf"
            ),
            "source_file": filename,
        }
        yield section


def create_search_index(index_client, index_name):
    if index_name not in index_client.list_index_names():
        index = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(
                    name="content", type="Edm.String", analyzer_name="en.microsoft"
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_configuration="default-vector-config",
                ),
                SimpleField(
                    name="category", type="Edm.String", filterable=True, facetable=True
                ),
                SimpleField(name="source_page", type="Edm.String", filterable=True),
                SimpleField(
                    name="source_file",
                    type="Edm.String",
                    filterable=True,
                ),
            ],
            vector_search=VectorSearch(
                algorithm_configurations=[
                    VectorSearchAlgorithmConfiguration(
                        name="default-vector-config",
                        kind="hnsw",
                        hnsw_parameters=HnswParameters(metric="cosine"),
                    )
                ]
            ),
        )
        index_client.create_index(index)
    else:
        print(f"Search index {index_name} already exists")


def index_sections(search_client, filename, sections):
    print(f"Indexing sections from '{filename}' into search index '{index_name}'")
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")


# Usage: python data_prep.py --data_input_dir ../data/input --data_output_dir ../data/output --category "handbook" # noqa
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", type=str, help="input document directory")
    parser.add_argument("--data_output_dir", type=str, help="output chunk directory")
    parser.add_argument("--category", type=str, help="category of the document")
    args = parser.parse_args()

    # check if output directory exists
    if not os.path.exists(args.data_output_dir):
        os.makedirs(args.data_output_dir)

    # load config
    config = dotenv_values("../.env")

    # Azure search service to use
    search_service = config["AZURE_SEARCH_SERVICE"]
    index_name = config["AZURE_SEARCH_INDEX"]
    search_cred = AzureKeyCredential(config["AZURE_SEARCH_KEY"])
    index_client = SearchIndexClient(
        endpoint=f"https://{config['AZURE_SEARCH_SERVICE']}.search.windows.net/",
        credential=search_cred,
    )
    search_client = SearchClient(
        endpoint=f"https://{search_service}.search.windows.net/",
        index_name=index_name,
        credential=search_cred,
    )

    # OpenAI for embedding
    openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
    openai.api_version = "2022-12-01"
    openai.api_type = "azure"
    openai.api_key = config["AZURE_OPENAI_API_KEY"]
    openai_embedding_model = config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    # select pdf files to process
    file_list = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.data_input_dir)
        for file in files
        if os.path.isfile(os.path.join(root, file)) and file.endswith(".pdf")
    ]

    # create Azure Cognitive Search index
    create_search_index(index_client, index_name)

    print("Start indexing files...")
    for file_path in file_list:
        # extract text from pdf
        page_map = get_document_text(file_path, args.data_output_dir)

        # create chunks and index them
        file_name = os.path.basename(file_path)
        sections = create_sections(
            file_name, page_map, args.data_output_dir, openai_embedding_model
        )
        index_sections(search_client, file_name, sections)
