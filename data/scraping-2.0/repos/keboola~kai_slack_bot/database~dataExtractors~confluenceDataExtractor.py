import os
import re
import logging
import json
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from atlassian import Confluence


openai.api_key=os.getenv('OPENAI_API_KEY')



class ConfluenceDataExtractor:
    def __init__(self, confluence_url, confluence_username, confluence_password, save_folder):
        self.confluence = Confluence(
            url=confluence_url, username=confluence_username, password=confluence_password
        )
        self.save_folder = save_folder

    def sanitize_filename(self, filename):
        return re.sub(r"[/\\]", "_", filename)

    def save_results(self, results, metadata, directory):
        for result in results:
            content_filename = os.path.join(
                directory, self.sanitize_filename(result["title"]) + ".txt"
            )
            metadata_filename = os.path.join(
                directory, self.sanitize_filename(result["title"]) + ".json"
            )

            html_content = result["body"]["storage"]["value"]
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            text = result["title"] + "\n\n" + text

            with open(content_filename, "w", encoding="utf-8") as file:
                file.write(text)

            with open(metadata_filename, "w", encoding="utf-8") as file:
                json.dump(metadata, file)

    def get_metadata(self, results):
        page_id = results[0].get("id")
        if page_id:
            data = self.confluence.get_page_by_id(page_id)

            page_metadata = {
                "id": data.get("id", ""),
                "CreatedDate": data["history"].get("createdDate", ""),
                "LastUpdatedDate": data["version"].get("when", ""),
                "Title": data.get("title", ""),
                "Creator": data["history"]["createdBy"].get("displayName", ""),
                "LastModifier": data["version"]["by"].get("displayName", ""),
                "url": data["_links"].get("base", "") + "/pages/" + data.get("id", ""),
                "Space": data["space"].get("name", ""),
            }

            return page_metadata
        return {}

    def download_confluence_pages(self, limit=100):
        spaces = self.confluence.get_all_spaces()
        for space in spaces.get("results"):
            logging.info(f"Downloading Confluence space: {space['name']}")

            content = self.confluence.get_space_content(space["key"])
            while True:
                subdir = os.path.join(self.save_folder, space["name"])
                os.makedirs(subdir, exist_ok=True)

                page = content.get("page")
                results = page.get("results")
                size = page.get("size")

                if not results:
                    logging.info(f"No results for {space['name']}")
                    break

                metadata = self.get_metadata(results)

                # Check if the document is already downloaded and up-to-date
                for result in results:
                    metadata_filename = os.path.join(
                        subdir, self.sanitize_filename(result["title"]) + ".json"
                    )

                    if os.path.exists(metadata_filename):
                        with open(metadata_filename, "r", encoding="utf-8") as file:
                            existing_metadata = json.load(file)
                            if (
                                 metadata["LastUpdatedDate"]
                                == existing_metadata.get("LastUpdatedDate")
                            ):
                                logging.info(
                                    f"Document '{result['title']}' is up-to-date. Skipping download."
                                )
                                continue

                self.save_results(results, metadata, subdir)

                if size == limit:
                    start = page.get("start") + limit
                    content = self.confluence.get_space_content(
                        space["key"], start=start, limit=limit
                    )
                    page = content.get("page")
                    results = page.get("results")
                    metadata = self.get_metadata(results)

                    # Check if the document is already downloaded and up-to-date
                    for result in results:
                        metadata_filename = os.path.join(
                            subdir, self.sanitize_filename(result["title"]) + ".json"
                        )

                        if os.path.exists(metadata_filename):
                            with open(metadata_filename, "r", encoding="utf-8") as file:
                                existing_metadata = json.load(file)
                                if (
                                    metadata["LastUpdatedDate"]
                                    == existing_metadata.get("LastUpdatedDate")
                                ):
                                    logging.info(
                                        f"Document '{result['title']}' is up-to-date. Skipping download."
                                    )
                                    continue

                    self.save_results(results, metadata, subdir)
                else:
                    break
