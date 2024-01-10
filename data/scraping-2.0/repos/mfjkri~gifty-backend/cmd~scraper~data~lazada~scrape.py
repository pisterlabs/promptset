import os
import json
import argparse
import time

import requests
import openai
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FOLDER_NAME = "1.rawData"
DESC_DATA_FOLDER_NAME = "2.descData"
DESC_ARCHIVE_DATA_FOLDER_NAME = "archive"
REFINED_DATA_FOLDER_NAME = "3.refinedData"
FORMATTED_DATA_FOLDER_NAME = "4.formattedData"
RAW_DATA_FOLDER = os.path.join(SCRIPT_DIR, RAW_DATA_FOLDER_NAME)
DESC_DATA_FOLDER = os.path.join(SCRIPT_DIR, DESC_DATA_FOLDER_NAME)
DESC_ARCHIVE_DATA_FOLDER = os.path.join(DESC_DATA_FOLDER, DESC_ARCHIVE_DATA_FOLDER_NAME)
REFINED_DATA_FOLDER = os.path.join(SCRIPT_DIR, REFINED_DATA_FOLDER_NAME)
FORMATTED_DATA_FOLDER = os.path.join(SCRIPT_DIR, FORMATTED_DATA_FOLDER_NAME)
openai.api_key = os.getenv("OPENAI_API_KEY")


def create_folder_structure(folder, search_param):
    folder_path = os.path.join(SCRIPT_DIR, folder, search_param)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# ---------------------------------------------------------------------------- #
#                                   FILL DESC                                  #
# ---------------------------------------------------------------------------- #
def extract_description(path, driver, wait):
    listing_url = "https://" + path
    driver.get(listing_url)

    time.sleep(1)
    html = wait.until(EC.presence_of_element_located((By.TAG_NAME, "html")))
    html.send_keys(Keys.PAGE_DOWN)
    ul_element = wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "pdp-product-desc"))
    )

    description_paragraph = ""
    for li_element in ul_element.find_elements(By.TAG_NAME, "li"):
        text_with_newlines = li_element.get_attribute("innerText")
        description_paragraph += text_with_newlines + "\n"
    return description_paragraph


def fill_description(category, results, driver, wait):
    for result in results:
        description = extract_description(result["item"]["itemUrl"][2::], driver, wait)
        result["description"] = description.rstrip()

    with open(
        os.path.join(DESC_DATA_FOLDER, f"{category}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open(
        os.path.join(DESC_ARCHIVE_DATA_FOLDER, f"{category}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                    REFINE                                    #
# ---------------------------------------------------------------------------- #
def generate_better_text(original_title, original_description):
    prompt = f"""
Title: {original_title}
Description: {original_description}

Generate an improved title and description to be used as a product listing on shopping websites.
This product description and title should be as generic as possible such that it should not be specific to the listing on Shopee.
It should not include any information that is not relevant to the product nor any quantity information.
Neither should it include things like "SG Stock" or "Local Seller" as it is not relevant to the product.
Product dimensions is okay in the description but not in the title.

Please write your answer in this format:
Title: <title>
Description: <description>
"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.5,
    )

    texts = response.choices[0].text.strip().split("\n")
    improved_texts = list(filter(None, texts))
    refined_title = improved_texts[0].removeprefix("Title: ")
    refined_description = improved_texts[1].removeprefix("Description: ")
    return refined_title, refined_description


def refine_data(filename):
    with open(os.path.join(DESC_DATA_FOLDER, filename), "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    refined_data = []
    for result in raw_data:
        refined_title, refined_description = generate_better_text(
            result["item"]["title"], result.get("description", "")
        )
        result["title"] = refined_title
        result["description"] = refined_description
        refined_data.append(result)

    with open(os.path.join(REFINED_DATA_FOLDER, filename), "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                    FORMAT                                    #
# ---------------------------------------------------------------------------- #
def format_data(filename):
    with open(os.path.join(REFINED_DATA_FOLDER, filename), "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = []
    for data in raw_data:
        listing = {
            "title": data["title"],
            "description": data["description"],
            "source": "https://" + data["item"]["image"][2::],
            "categories": [filename.removesuffix(".json")],
            "price": float(
                data["item"]["sku"]["def"]["price"]
                if data["item"]["sku"]["def"]["price"]
                else data["item"]["sku"]["def"]["promotionPrice"]
            ),
            "platform": "Lazada",
            "purchaseUrl": "https://" + data["item"]["itemUrl"][2::],
        }
        formatted_data.append(listing)

    with open(
        os.path.join(FORMATTED_DATA_FOLDER, filename), "w", encoding="utf-8"
    ) as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Lazada Scraper")
    parser.add_argument(
        "action",
        choices=["populate", "filldesc", "refine", "format"],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "populate":
        url = os.getenv("LAZADA_RAPID_API_URL")
        categories = [
            # "Electronics and Gadgets",
            # "Home and Kitchen",
            # "Fashion and Accessories",
            # "Books and Stationery",
            # "Beauty and Personal Care",
            # "Toys and Games",
            # "Sports and Outdoor Gear",
            # "Art and Craft Supplies",
            # "Food and Gourmet Gifts",
            # "Travel and Adventure",
        ]

        headers = {
            "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
            "X-RapidAPI-Host": os.getenv("LAZADA_RAPID_API_HOST"),
        }

        for category in categories:
            for page_number in range(1, 2):
                querystring = {"q": category, "p": page_number, "region": "SG"}
                response = requests.get(url, headers=headers, params=querystring)

                if response.status_code == 200:
                    folder_path = create_folder_structure(RAW_DATA_FOLDER, category)
                    file_path = os.path.join(folder_path, f"{page_number}.json")

                    with open(file_path, "w") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=4)

                    print(f"Saved response for '{category}' page {page_number}")
                else:
                    print(
                        f"Error for '{category}' page {page_number}: {response.status_code}"
                    )

    elif args.action == "filldesc":
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        options.page_load_strategy = "eager"
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 10)

        os.makedirs(DESC_DATA_FOLDER, exist_ok=True)
        os.makedirs(DESC_ARCHIVE_DATA_FOLDER, exist_ok=True)
        for category in os.listdir(RAW_DATA_FOLDER):
            category_path = os.path.join(RAW_DATA_FOLDER, category)
            if os.path.isdir(category_path) and not os.path.exists(
                os.path.join(DESC_DATA_FOLDER, f"{category}.json")
            ):
                combined_results = []
                for filename in os.listdir(category_path):
                    if filename.endswith(".json"):
                        json_path = os.path.join(category_path, filename)
                        with open(json_path, "r") as json_file:
                            json_data = json.load(json_file)
                            combined_results.extend(json_data["result"]["resultList"])

                fill_description(category, combined_results, driver, wait)
                print(f"Extracted description for {category}")

        driver.quit()

    elif args.action == "refine":
        os.makedirs(REFINED_DATA_FOLDER, exist_ok=True)
        for filename in os.listdir(DESC_DATA_FOLDER):
            if filename.endswith(".json") and not os.path.exists(
                os.path.join(REFINED_DATA_FOLDER, filename)
            ):
                refine_data(filename)
                print(f"Generated better descriptions for {filename}")

    elif args.action == "format":
        os.makedirs(FORMATTED_DATA_FOLDER, exist_ok=True)
        for filename in os.listdir(REFINED_DATA_FOLDER):
            if filename.endswith(".json"):
                format_data(filename)
                print(f"Formatted {filename}")

    else:
        print("Invalid action")


if __name__ == "__main__":
    main()
