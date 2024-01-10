from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import os
import asyncio
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from newspaper import Article
from typing import List  # Import the List type hint
import aiohttp
from main import createPost


class ArxivPostCreator:
    def __init__(self, output_dir="output", semaphores=5):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "arxiv_used_dois.txt")
        self.processed_dois = set()
        self.semaphore = asyncio.Semaphore(semaphores)
        with open("token.txt", "r") as f:
            self.token = f.read()

        response_schemas = [
            ResponseSchema(
                name="title",
                description="Generate a precise title that captures the essence of the information in keywords. Make it understadable by broad audience and very short. Can be as short as one word.",
            ),
            ResponseSchema(
                name="description",
                description="Elaborate on the title with a short, one-sentence description.",
            ),
            ResponseSchema(
                name="search_query",
                description="Generate a search query for google images to find the most representative picture of the exact information.",
            ),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )

        format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template="Summary should be in an unbiased manner. Create an understandable title and short description of the information\n{format_instructions}\n{information}",
            # include - to keep it understandable and not use complex words, and write as neutral as possible
            input_variables=["information"],
            partial_variables={"format_instructions": format_instructions},
        )

    async def load_processed_dois(self):
        if not os.path.exists(self.output_file):
            # Create the file if it doesn't exist
            with open(self.output_file, "w"):
                pass
        with open(self.output_file, "r") as f:
            lines = f.read().splitlines()
            self.processed_dois.update(map(str, lines))

    async def save_processed_dois(self):
        with open(self.output_file, "w") as f:
            f.write("\n".join(map(str, self.processed_dois)))

    async def fetch_arxiv_data(self, category, search_query, start=0, max_results=10):
        url = f"http://export.arxiv.org/api/query?search_query={category}:{search_query}&start={start}&max_results={max_results}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                feed_content = await response.text()

        feed = feedparser.parse(feed_content)

        processed = []

        for entry in feed.entries:
            doi = entry.get("id", None)
            if doi and doi not in self.processed_dois:
                title = entry.get("title", "")
                summary = entry.get("summary", "")

                # Get the link to the PDF
                pdf_link = ""
                for link in entry.get("links", []):
                    if link.get("type") == "application/pdf":
                        pdf_link = link.get("href")
                        break

                # Store the information in a dictionary
                post_info = {
                    "doi": doi,
                    "title": title,
                    "summary": summary,
                    "pdf_link": pdf_link,
                }
                print(post_info)

                processed.append(post_info)
        return processed

    async def process_arxiv_post(self, post_info):
        async with self.semaphore:
            doi = post_info["doi"]
            title = post_info["title"]
            summary = post_info["summary"]
            pdf_link = post_info["pdf_link"]
            text = f"{title}\n{summary}"
            links = [pdf_link]
            if len(text) < 120:
                return None  # Skip processing if the content is too short

            image_url = None  # await extract_square_image_from_pdf(pdf_link, 200)
            err, response, data = await createPost(
                self.prompt, self.output_parser, text, self.token, links, image_url
            )
            if err:
                return
            if response.status == 200:
                self.processed_dois.add(doi)  # Mark the post as processed
                await self.save_processed_dois()  # Save the updated list of processed IDs
                print("Success:", data)
            else:
                print(f"{response.status} Server error:", data)

    async def run(self):
        await self.load_processed_dois()
        posts = await self.fetch_arxiv_data("all", "machine learning", 0, 10)

        # Create tasks to fetch details for each story concurrently
        tasks = [self.process_arxiv_post(post_info) for post_info in posts]
        await asyncio.gather(*tasks)
