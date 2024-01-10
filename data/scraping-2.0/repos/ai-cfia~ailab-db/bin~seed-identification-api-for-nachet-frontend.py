import os
import json
import logging
import sys

import ailab.db as db
import ailab.db.nachet as nachet
from ailab.models import openai

from ailab.db.nachet.seed_queries import seeds_urls
from ailab.db.nachet.seed_queries import get_seed_name
from ailab.db.nachet.seed_queries import get_webpage
from ailab.db.nachet.seed_queries import get_images

logging.basicConfig(
    filename="mylog.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
WEBSITE_URL = "https://inspection.canada.ca"


def create_seed_url_mapping(cursor, list_seed_url):
    ### Get a name from the seed URL
    url_to_seed_mapping = {}

    for rows in list_seed_url:
        seed_full_url = WEBSITE_URL + rows["seeds_url"]
        seed_name_query = get_seed_name(cursor, seed_full_url)

        if seed_name_query:
            seed_name = seed_name_query[0]["sd_nme"]
            url_to_seed_mapping[seed_full_url] = seed_name
    return url_to_seed_mapping


def transform_seed_data_into_json(
    cursor,
    url_to_seed_mapping,
    system_prompt,
    load_user_prompt,
    json_template,
    seed_data_path,
):
    """
    Process seed data using Azure OpenAI endpoint and save results as JSON files.

    Args:
        system_prompt (str): A system prompt for the OpenAI conversation.
        user_prompt (str): A user prompt for the OpenAI conversation.
        json_template (json): A JSON template for the OpenAI request.

    This function performs the following steps:
    1. Iterates through a list of seed values.
    2. Checks if a JSON file for each seed exists and skips if it does.
    3. Constructs an SQL query to retrieve data related to the seed from a database.
    4. Sends the query to the database and fetches the retrieved data.
    5. Concatenates the cleaned content into a single 'page.'
    6. Sends a request to the Azure OpenAI endpoint to get a response.
    7. Processes the response, extracting the name and saving it as a JSON file.
    """
    for url, seed_name in url_to_seed_mapping.items():
        logging.info("Current seed: %s", seed_name)

        seed_json_path = seed_name + ".json"

        file_path = os.path.join(seed_data_path, seed_json_path)

        if os.path.exists(file_path):
            logging.info(
                "JSON file %s exists in %s, skipping", seed_json_path, seed_data_path
            )
        else:
            web_pages = get_webpage(cursor, url)

            all_language_seed_page = ""
            for row in web_pages:
                web_text = row.get("cleaned_content")
                all_language_seed_page += web_text
            page = all_language_seed_page
            md5hash = row.get("md5hash")

            ### Get the images corresponding to the current page
            images_fetch = get_images(cursor, md5hash)

            image_information = ""

            for row in images_fetch:
                image_links = row["photo_link"]
                image_descriptions = row["photo_description"]
                image_information += f"Image link: {image_links}"
                image_information += f"\nImage description: {image_descriptions}\n\n"

            logging.info("Sending request for summary to Azure OpenAI endpoint...\n")

            user_prompt = (
                load_user_prompt
                + "Return a JSON file that follows this template:\n\n"
                + json_template
                + "\n\nhere is the text to parse:\n"
                + page
                + "\n\nhere is the source url of the page:\n"
                + url
                + "\n\nAnd here is the images descriptions:\n"
                + image_information
            )

            response = openai.get_chat_answer(system_prompt, user_prompt, 2000)

            data = json.loads(response.choices[0].message.content)

            if isinstance(data, dict):
                file_name = seed_name
                file_name = file_name.encode("latin1").decode("unicode-escape")
                file_name += ".json"

                file_path = os.path.join(seed_data_path, file_name)
                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)

                logging.info("JSON data written to %s", file_path)
            else:
                logging.error(
                    "Error: not a dictionary, so it cannot be serialized to JSON."
                )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " SEED_DATA_PATH PROMPT_PATH")
        print("SEED_DATA_PATH: Directory for storing seeds")
        print("PROMPT_PATH: Directory containing the API prompt")
        sys.exit(1)

    SEED_DATA_PATH = sys.argv[1]
    PROMPT_PATH = sys.argv[2]

    if not os.path.exists(SEED_DATA_PATH):
        print(f"The directory '{SEED_DATA_PATH}' does not exist.")
        sys.exit(1)

    if not os.path.exists(PROMPT_PATH):
        print(f"The directory '{PROMPT_PATH}' does not exist.")
        sys.exit(1)

    system_prompt = nachet.load_prompt(PROMPT_PATH, "system_prompt.txt")
    load_user_prompt = nachet.load_prompt(PROMPT_PATH, "user_prompt.txt")
    json_template = nachet.load_json_template(PROMPT_PATH)

    nachet_db = db.connect_db()
    with nachet_db.cursor() as cursor:
        seed_urls = seeds_urls(cursor, 10)
        url_to_seed_mapping = create_seed_url_mapping(cursor, seed_urls)
        logging.info("%s", url_to_seed_mapping)

        logging.info("\nList of selected seeds :")
        for url, seed_name in url_to_seed_mapping.items():
            logging.info("%s", seed_name)

        transform_seed_data_into_json(
            cursor,
            url_to_seed_mapping,
            system_prompt,
            load_user_prompt,
            json_template,
            SEED_DATA_PATH,
        )
