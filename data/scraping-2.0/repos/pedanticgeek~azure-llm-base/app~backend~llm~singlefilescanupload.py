import os
import fitz
import time
from typing import Any, List, Dict
import tiktoken
from openai import RateLimitError

from utils import filename_to_id, encode_image
from llm.assistants import get_or_create_assistant_by_name, page_scanning_template
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


class SingleFileScanUpload:
    """
    The logic:
    1. Splits the document into pages and saves them as images using fitz (PyMuPDF)
    2. Uses GPT-4 Vision model to scan the pages for informative data
    3. Uploads the source file, pages, and texts into Azure Blob Storage
    4. Uses summary assistant to summarize the document
    5. Indexes the outputs in Azure Cognitive Search

    This class is only used in queued tasks.
    """

    def __init__(self):
        self.summary_assistant = get_or_create_assistant_by_name(
            gpt.CHATGPT_MODEL, "document-summarization"
        )
        self.temp_dir = "temp"

    def calculate_tokens(self, input: str):
        encoding = tiktoken.encoding_for_model(gpt.CHATGPT_MODEL)
        return len(encoding.encode(input))

    def download_source_file(self, filename):
        logger.info(f"Downloading source file '{filename}' to Azure Blob Storage")
        result = az.blob_container.download_blob(f"sourcefiles/{filename}")
        return result

    def get_temp_image_path(self, filename, page):
        return f"{self.temp_dir}/{filename}-{page}.png"

    def upload_blobs(self, filename, page_map: List[Dict[str, Any]], file_id):
        for page in page_map:
            blob_name = f"{filename}-page{page['page_num']}"
            image_path = self.get_temp_image_path(filename, page["page_num"])
            # Step 1. Upload image
            with open(image_path, "rb") as f:
                az.blob_container.upload_blob(
                    blob_name + ".png",
                    f.read(),
                    overwrite=True,
                    metadata={"id": file_id},
                )
            # Step 2. Upload text
            az.blob_container.upload_blob(
                blob_name + ".txt",
                page["page_text"],
                overwrite=True,
                metadata={"id": file_id},
            )

    def split_pdf_into_images(self, filename: str, content) -> List[str]:
        """This method saves pdf pages as images in the temp directory"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        doc = fitz.open(filename, content.read())

        # Iterate through each page
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load each page
            pix = page.get_pixmap(dpi=150)  # Render page to an image pixmap
            image_path = self.get_temp_image_path(filename, page_num)
            images.append({"page_num": page_num, "image_file": image_path})
            pix.save(image_path)  # Save the image of the page

        doc.close()
        images.sort(key=lambda x: x["page_num"])
        return images

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=15, max=60),
        stop=stop_after_attempt(30),
        before_sleep=logger.vision_limit_reached,
    )
    def scan_page_image(self, image_file: str) -> str:
        """
        As of the moment of writing this code, OpenAI only has GPT4-Vision in preview and it's only available through ChatCompletion API.
        """
        logger.info(f"Scanning page image '{image_file}' using OpenAI")
        base64_image = encode_image(image_file)
        payload = {
            "model": gpt.CHATGPT_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": page_scanning_template},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2048,
        }
        response = gpt.client.chat.completions.create(**payload)
        logger.info(f"Output: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def scan_page_images(self, image_files: List[Dict[int, str]]):
        page_map = []
        offset = 0
        for img in image_files:
            image_text = self.scan_page_image(img["image_file"])
            page_map.append(
                {
                    "page_num": img["page_num"],
                    "page_offset": offset,
                    "page_text": image_text,
                }
            )
            offset += len(image_text)
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

    @staticmethod
    def split_text(page_map, filename):
        """This method splits pages of the document into sections that later will be stored in Azure Cognitive Search"""
        SENTENCE_ENDINGS = [".", "!", "?"]
        WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
        logger.info(f"Splitting '{filename}' into sections")

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
        self,
        title,
        category,
        filename,
        page_map,
        file_id,
        is_summary=False,
        is_assessment=False,
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
                "sourcepage": f"{filename}-page{i}.txt",
                "sourcefile": filename,
                "is_summary": is_summary,
                "is_assessment": is_assessment,
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

    def add_message_to_thread(self, page, thread):
        gpt.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=page["page_text"],
            metadata={"page_num": page["page_num"]},
        )

    def run_openai_assistant(self, filename, page_map):
        """
        This method runs OpenAI assistants
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

        # Step 3: Add summary
        logger.info(f"Summarizing '{filename}' using OpenAI")
        run = gpt.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.summary_assistant.id,
        )

        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = gpt.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
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
        images = self.split_pdf_into_images(filename, file)
        try:
            # Step 2. Uses GPT-4 Vision model to scan the pages for informative data
            page_map = self.scan_page_images(images)

            # Step 3. Upload pages into Azure Blob Storage
            self.upload_blobs(filename, page_map, file_id)

            # Step 4. Run the summarization
            summary = self.run_openai_assistant(filename, page_map)

            # Step 5. Index pages into Azure Cognitive Search
            await self.index_document(
                title=summary["title"],
                category="Business Summary Document",
                filename=filename,
                file_id=file_id,
                page_map=page_map,
            )

            # And summary
            await self.index_document(
                title=summary["title"],
                category="Business Summary Document",
                filename=filename,
                file_id=file_id,
                page_map=[
                    {"page_num": 0, "page_offset": 0, "page_text": summary["summary"]}
                ],
                is_summary=True,
            )

            # Cleanup
            for i in os.listdir(self.temp_dir):
                os.remove(f"{self.temp_dir}/{i}")

            return {**summary, "filename": filename, "id": file_id}
        except Exception as e:
            # Cleanup
            for i in os.listdir(self.temp_dir):
                os.remove(f"{self.temp_dir}/{i}")
            raise e
