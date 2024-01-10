import os
import html
import time
from typing import Any, List, Dict, Optional
import tiktoken
from openai import RateLimitError
from utils import filename_to_id
from llm.assistants import get_or_create_assistant_by_name
from config import logger, az, gpt
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100


class SingleFileUpload:
    """
    The logic:
    1. Recognizes the document using Azure Form Recognizer and splits it into pages
    2. Upload pages into Azure Blob Storage
    3. Summarize the file using OpenAI
    4. Upload pages to Azure Cognitive Search
    5. Upload summary to Azure Cognitive Search

    This class is only used in queued tasks.
    """

    def __init__(self):
        self.assistant = get_or_create_assistant_by_name(
            gpt.CHATGPT_MODEL, "document-summarization"
        )

    def calculate_tokens(self, input: str):
        encoding = tiktoken.encoding_for_model(gpt.CHATGPT_MODEL)
        return len(encoding.encode(input))

    @staticmethod
    def blob_name_from_file_page(filename, page=0):
        filename_, _ = filename.split(".")
        return f"{filename_}-page{page}.txt"

    def download_source_file(self, filename):
        logger.info(f"Downloading source file '{filename}' to Azure Blob Storage")
        return az.blob_container.download_blob(f"sourcefiles/{filename}")

    def upload_blobs(self, filename, pages: List[Dict[str, Any]], file_id):
        for page in pages:
            blob_name = self.blob_name_from_file_page(filename, page["page_num"])
            logger.info(f"\tUploading blob for page {page['page_num']} -> {blob_name}")
            az.blob_container.upload_blob(
                blob_name, page["page_text"], overwrite=True, metadata={"id": file_id}
            )

    @staticmethod
    def table_to_html(table):
        table_html = "<table>"
        rows = [
            sorted(
                [cell for cell in table.cells if cell.row_index == i],
                key=lambda cell: cell.column_index,
            )
            for i in range(table.row_count)
        ]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = (
                    "th"
                    if (cell.kind == "columnHeader" or cell.kind == "rowHeader")
                    else "td"
                )
                cell_spans = ""
                if cell.column_span > 1:
                    cell_spans += f" colSpan={cell.column_span}"
                if cell.row_span > 1:
                    cell_spans += f" rowSpan={cell.row_span}"
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html += "</tr>"
        table_html += "</table>"
        return table_html

    def get_document_text(self, filename, content):
        offset = 0
        page_map = []
        logger.info(f"Extracting text from '{filename}' using Azure Form Recognizer")
        poller = az.form_recognizer.begin_analyze_document(
            "prebuilt-layout", document=content
        )
        form_recognizer_results = poller.result()
        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [
                table
                for table in form_recognizer_results.tables
                if table.bounding_regions[0].page_number == page_num + 1
            ]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1] * page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >= 0 and idx < page_length:
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
            page_map.append(
                {"page_num": page_num, "page_offset": offset, "page_text": page_text}
            )
            offset += len(page_text)

        return page_map

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=15, max=60),
        stop=stop_after_attempt(15),
        before_sleep=logger.embedding_limit_reached,
    )
    def compute_embedding(self, content):
        res = gpt.client.embeddings.create(model=gpt.EMB_MODEL_NAME, input=content)
        return res.data[0].embedding

    def add_message_to_thread(self, page, thread):
        gpt.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=page["page_text"],
            metadata={"page_num": page["page_num"]},
        )

    @staticmethod
    def split_text(page_map, filename):
        """This method splits pages of the document into sections that later will be stored in Azure Cognitive Search"""
        SENTENCE_ENDINGS = [".", "!", "?"]
        WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
        logger.info(f"Splitting {filename} into sections")

        def find_page(offset):
            num_pages = len(page_map)
            for i in range(num_pages - 1):
                if (
                    offset >= page_map[i]["page_offset"]
                    and offset < page_map[i + 1]["page_offset"]
                ):
                    return i
            return num_pages - 1

        all_text = "".join(p["page_text"] for p in page_map)
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
                if (
                    end < length
                    and all_text[end] not in SENTENCE_ENDINGS
                    and last_word > 0
                ):
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
                # If the section ends with an unclosed table, we need to start the next section with the table.
                # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
                # If last table starts inside SECTION_OVERLAP, keep overlapping
                logger.info(
                    f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}"
                )
                start = min(end - SECTION_OVERLAP, start + last_table_start)
            else:
                start = end - SECTION_OVERLAP

        if start + SECTION_OVERLAP < end:
            yield (all_text[start:end], find_page(start))

    async def index_document(
        self, title, category, filename, file_id, page_map, is_summary=False
    ) -> str:
        """
        This method indexes a document into Azure Cognitive Search
        """
        logger.info(
            f"Indexing sections from '{filename}' into search index '{az.SEARCH_INDEX}'"
        )
        # Step 1: Generate sections
        sections = []
        for i, (content, pagenum) in enumerate(self.split_text(page_map, filename)):
            section = {
                "id": f"{file_id}-page-{i}",
                "content": content,
                "title": title,
                "category": category,
                "sourcepage": self.blob_name_from_file_page(filename, pagenum),
                "sourcefile": filename,
                "is_summary": is_summary,
            }
            section["embedding"] = self.compute_embedding(content)
            sections.append(section)
        # Step 2. Index Sections
        i = 0
        batch = []
        # Preventive save every 1000 sections
        for s in sections:
            batch.append(s)
            i += 1
            if i % 1000 == 0:
                results = await az.search_client.upload_documents(documents=batch)
                succeeded = sum([1 for r in results if r.succeeded])
                logger.info(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
                batch = []
        # Save the remaining sections
        if len(batch) > 0:
            results = await az.search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            logger.info(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

    def run_openai_assistant(self, filename, page_map):
        """
        This method summarizes a document using OpenAI
        """
        logger.info(f"Summarizing '{filename}' using OpenAI")
        # Step 1: Create a thread
        thread = gpt.client.beta.threads.create(
            metadata={
                "filename": filename,
                "pages_count": len(page_map),
            }
        )
        # Step 2: Insert every page as a message
        for page in page_map:
            self.add_message_to_thread(page, thread)

        # Step 3: Start run
        run = gpt.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        )

        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = gpt.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

        # Step 4: Get the result
        thread_messages = gpt.client.beta.threads.messages.list(thread.id)
        res = thread_messages.data[0].content[0].text.value
        summary = eval(res[res.find("{") : res.rfind("}") + 1])
        logger.info(f"Title: {summary['title']}")
        logger.info(f"Category: {summary['category']}")
        logger.info(f"Summary: {summary['summary']}")
        gpt.client.beta.threads.delete(thread.id)
        return summary

    async def run(self, filename: str) -> None:
        file_id = filename_to_id(filename)
        # Download the file
        file = self.download_source_file(filename)
        # Step 1. Recognize the document using Azure Form Recognizer and split it into pages
        page_map = self.get_document_text(filename, file)

        # Step 2. Upload pages into Azure Blob Storage
        self.upload_blobs(filename, page_map, file_id)

        # Step 3. Summarize the file using OpenAI
        summary = self.run_openai_assistant(filename, page_map)

        # Step 4. Index pages into Azure Cognitive Search
        await self.index_document(
            title=summary["title"],
            category=summary["category"],
            filename=filename,
            file_id=file_id,
            page_map=page_map,
        )

        # Step 5. Index summary into Azure Cognitive Search
        await self.index_document(
            title=summary["title"],
            category=summary["category"],
            filename=filename,
            file_id=file_id,
            page_map=[
                {"page_num": 0, "page_offset": 0, "page_text": summary["summary"]}
            ],
            is_summary=True,
        )

        return {**summary, "filename": filename, "id": file_id}
