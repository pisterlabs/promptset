import base64
import html
import io
import os
import re
import time

import openai
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_random_exponential

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = 'openai_token_cred'
CACHE_KEY_CREATED_TIME = 'created_time'
CACHE_KEY_TOKEN_TYPE = 'token_type'


class CogIndexer:

    open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
    #open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = azd_credential
    open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] = "azure_ad"

    def __init__(self, args:dict):
        self.args = args


    def blob_name_from_file_page(self, filename, page = 0):
        if os.path.splitext(filename)[1].lower() == ".pdf":
            return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
        else:
            return os.path.basename(filename)

    def upload_blobs(self, filename, file_contents):
        blob_service = BlobServiceClient(account_url=f"https://{os.getenv('STORAGEACCOUNT')}.blob.core.windows.net", credential=self.args["storage_creds"])
        blob_container = blob_service.get_container_client(os.getenv('CONTAINER'))
        if not blob_container.exists():
            blob_container.create_container()

        try:
            # if file is PDF split into pages and upload each page as a separate blob
            if os.path.splitext(filename)[1].lower() == ".pdf":

                reader = PdfReader(file_contents)
                pages = reader.pages
                for i in range(len(pages)):
                    blob_name = self.blob_name_from_file_page(filename, i)
                    #if arg_s.verbose: print(f"\tUploading blob for page {i} -> {blob_name}")
                    f = io.BytesIO()
                    writer = PdfWriter()
                    writer.add_page(pages[i])
                    writer.write(f)
                    f.seek(0)
                    blob_container.upload_blob(blob_name, f, overwrite=True)
            else:
                blob_name = self.blob_name_from_file_page(filename)
                with open(filename,"rb") as data:
                    blob_container.upload_blob(blob_name, data, overwrite=True)
        except Exception as e:
            print(f"Error uploading blob for '{filename}': {e}")

    def remove_blobs(self, filename):
        #if arg_s.verbose: print(f"Removing blobs for '{filename or '<all>'}'")
        blob_service = BlobServiceClient(account_url=f"https://{os.getenv('STORAGEACCOUNT')}.blob.core.windows.net", credential=self.args["storage_creds"])
        blob_container = blob_service.get_container_client(os.getenv('CONTAINER'))
        if blob_container.exists():
            if filename is None:
                blobs = blob_container.list_blob_names()
            else:
                prefix = os.path.splitext(os.path.basename(filename))[0]
                blobs = filter(lambda b: re.match(f"{prefix}-\d+\.pdf", b), blob_container.list_blob_names(name_starts_with=os.path.splitext(os.path.basename(prefix))[0]))
            for b in blobs:
                #if arg_s.verbose: print(f"\tRemoving blob {b}")
                blob_container.delete_blob(b)

    def table_to_html(self, table):
        table_html = "<table>"
        rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
                cell_spans = ""
                if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
                if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html +="</tr>"
        table_html += "</table>"
        return table_html

    def get_document_text(self, filename):
        offset = 0
        page_map = []
        if self.args["localpdfparser"]:
            reader = PdfReader(filename)
            pages = reader.pages
            for page_num, p in enumerate(pages):
                page_text = p.extract_text()
                page_map.append((page_num, offset, page_text))
                offset += len(page_text)
        else:
            #if arg_s.verbose: print(f"Extracting text from '{filename}' using Azure Form Recognizer")
            form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{os.getenv('FORMRECOGNIZERSERVICE')}.cognitiveservices.azure.com/", credential=self.args['formrecognizer_creds'], headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
            #with open(filename, "rb") as f:
            f = filename.getvalue()
            try:
                poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
            except Exception as e:
                print(f"Error analyzing document '{filename}': {e}")
                return page_map
            
            form_recognizer_results = poller.result()

            for page_num, page in enumerate(form_recognizer_results.pages):
                tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

                # mark all positions of the table spans in the page
                page_offset = page.spans[0].offset
                page_length = page.spans[0].length
                table_chars = [-1]*page_length
                for table_id, table in enumerate(tables_on_page):
                    for span in table.spans:
                        # replace all table spans with "table_id" in table_chars array
                        for i in range(span.length):
                            idx = span.offset - page_offset + i
                            if idx >=0 and idx < page_length:
                                table_chars[idx] = table_id

                # build page text by replacing characters in table spans with table html
                page_text = ""
                added_tables = set()
                for idx, table_id in enumerate(table_chars):
                    if table_id == -1:
                        page_text += form_recognizer_results.content[page_offset + idx]
                    elif table_id not in added_tables:
                        page_text += self.table_to_html(tables_on_page[table_id])
                        added_tables.add(table_id)

                page_text += " "
                page_map.append((page_num, offset, page_text))
                offset += len(page_text)

        return page_map

    def split_text(self, page_map):
        SENTENCE_ENDINGS = [".", "!", "?"]
        WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
        #if arg_s.verbose: print(f"Splitting '{filename}' into sections")

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
                while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                    if all_text[end] in WORDS_BREAKS:
                        last_word = end
                    end += 1
                if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                    end = last_word # Fall back to at least keeping a whole word
            if end < length:
                end += 1

            # Try to find the start of the sentence or at least a whole word boundary
            last_word = -1
            while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
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
            if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
                # If the section ends with an unclosed table, we need to start the next section with the table.
                # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
                # If last table starts inside SECTION_OVERLAP, keep overlapping
                #if arg_s.verbose: print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
                start = min(end - SECTION_OVERLAP, start + last_table_start)
            else:
                start = end - SECTION_OVERLAP

        if start + SECTION_OVERLAP < end:
            yield (all_text[start:end], find_page(start))

    def filename_to_id(self, filename):
        filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
        filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
        return f"file-{filename_ascii}-{filename_hash}"

    def create_sections(self, filename, page_map, use_vectors):
        file_id = self.filename_to_id(filename)
        for i, (content, pagenum) in enumerate(self.split_text(page_map)):
            section = {
                "id": f"{file_id}-page-{i}",
                "content": content,
                "category": self.args["category"],
                "sourcepage": self.blob_name_from_file_page(filename, pagenum),
                "sourcefile": filename
            }

            if use_vectors:
                section["embedding"] = self.compute_embedding(content)
            yield section

    def before_retry_sleep(self, retry_state):
        print("Rate limited on the OpenAI embeddings API, sleeping before retrying...")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
    def compute_embedding(self, text):
        #self.refresh_openai_token()
        return openai.Embedding.create(engine=os.getenv('OPENAIDEPLOYMENT'), input=text)["data"][0]["embedding"]

    def create_search_index(self):
        #if arg_s.verbose: print(f"Ensuring search index {self.args["index} exists")
        index_client = SearchIndexClient(endpoint=f"https://{os.getenv('SEARCHSERVICE')}.search.windows.net/",
                                        credential=self.args["search_creds"])
        if self.args["index"] not in index_client.list_index_names():
            index = SearchIndex(
                name=self.args["index"],
                fields=[
                    SimpleField(name="id", type="Edm.String", key=True),
                    SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft"),
                    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                                vector_search_dimensions=1536, vector_search_configuration="default"),
                    SimpleField(name="category", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True)
                ],
                semantic_settings=SemanticSettings(
                    configurations=[SemanticConfiguration(
                        name='default',
                        prioritized_fields=PrioritizedFields(
                            title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))]),
                    vector_search=VectorSearch(
                        algorithm_configurations=[
                            VectorSearchAlgorithmConfiguration(
                                name="default",
                                kind="hnsw",
                                hnsw_parameters=HnswParameters(metric="cosine")
                            )
                        ]
                    )
                )
            #if arg_s.verbose: print(f"Creating {self.args["index} search index")
            index_client.create_index(index)
        else:
            print(f"Search index {self.args['index']} already exists")

    def index_sections(self, filename, sections):
        #if arg_s.verbose: print(f"Indexing sections from '{filename}' into search index '{self.args["index}'")
        search_client = SearchClient(endpoint=f"https://{os.getenv('SEARCHSERVICE')}.search.windows.net/",
                                        index_name=self.args["index"],
                                        credential=self.args["search_creds"])
        i = 0
        batch = []
        if len(batch) == 0:
            print('nothing in batch'    )
        
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


    def remove_from_index(self, filename):
        print(f"Removing sections from '{filename or '<all>'}' from search index '{self.args['index']}'")
        search_client = SearchClient(endpoint=f"https://{os.getenv('SEARCHSERVICE')}.search.windows.net/",
                                        index_name=self.args["index"],
                                        credential=self.args["search_creds"])
        while True:
            filter = None if filename is None else f"sourcefile eq '{os.path.basename(filename)}'"
            r = search_client.search("", filter=filter, top=1000, include_total_count=True)
            if r.get_count() == 0:
                break
            r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
            #if arg_s.verbose: print(f"\tRemoved {len(r)} sections from index")
            # It can take a few seconds for search results to reflect changes, so wait a bit
            time.sleep(2)

    # refresh open ai token every 5 minutes
    def refresh_openai_token():
        if open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] == 'azure_ad' and open_ai_token_cache[CACHE_KEY_CREATED_TIME] + 300 < time.time():
            token_cred = open_ai_token_cache[CACHE_KEY_TOKEN_CRED]
            openai.api_key = token_cred.get_token("https://cognitiveservices.azure.com/.default").token
            open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
        else:
            print('dont need one')