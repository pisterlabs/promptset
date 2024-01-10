from pyairtable import Table
import os
import logging
import json
import openai
from easygpt import EasyGPT # https://github.com/chubajs/easygpt

logging.basicConfig(filename="logs/classify.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
logging.info("Starting classify.py")

TOKENS_FOR_16K = 3300
TOKENS_TO_CANCEL = 10000

# Initialize static prompts
CLASSIFY_PROMPT = "Classify following article:\n\n[article]"
CLASSIFY_SYSTEM_MESSAGE = """
You are AI parser who parses content to classify it. You always respond with JSON. Your aim is to make summary for the article which is 4-5 sentences long, 4 tags for the article as a list and countries if any mentioned (if cities mentioned and you know the countries then add countries of the city). Also select if any categories can apply to this article from this limited list:

travel, motivation, religion, music, tv, health, food, catering, tech, marketing, ai, content, general

select up to 3, use only this categories.

Here is an example of your response:

{
  "summary": "The article delves into the booming market for plant-based food alternatives in Berlin, Germany. It discusses how technology like 3D printing is innovating meat substitutes, the health benefits of shifting to a plant-based diet, and why Berlin has become the hub for this growing industry. Local startups are attracting global investment, positioning Berlin as a leading city in sustainable food technology.",
  "tags": ["Health", "Tech", "Food", "Sustainability"],
  "countries": ["Germany"],
  "categories": ["Health", "Tech", "Food"]
}

and one more example:

{
  "summary": "The article explores the impact of Japanese minimalist design on contemporary marketing strategies. It highlights how the principle of 'Ma' (the space in between) can be leveraged in advertising for stronger emotional impact. Case studies from renowned brands like Apple and Uniqlo are dissected to demonstrate this influence. The article argues that adopting minimalist design leads to clearer messaging and greater consumer engagement.",
  "tags": ["Marketing", "Design", "Culture", "Consumer Engagement"],
  "countries": ["Japan", "United States"],
  "categories": ["Marketing", "Content"]
}
"""

# Set API keys from environment variables
api_key = os.environ["AIRTABLE_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize Airtable
base_id = "applCPBBlTlF2v1NE"
table_news = Table(api_key, base_id, "News")

# Initialize EasyGPT
gpt = EasyGPT(openai, "gpt-3.5-turbo")



# Fetch news without summary
news_to_classify = table_news.all(formula="AND({summary}='')")

logging.info(f"Found {len(news_to_classify)} news to classify")
print(f"Found {len(news_to_classify)} news to classify")

# Loop to classify each article
for article in news_to_classify:

    if "content" not in article["fields"]:
        continue

    content = article["fields"]["content"]
    tokens = gpt.tokens_in_string(content)

    logging.info(f"Classifying {article['fields']['title']}, tokens: {tokens}")
    print(f"Classifying {article['fields']['title']}, tokens: {tokens}")

    if tokens > TOKENS_TO_CANCEL:
        table_news.update(article["id"], {"errors": article["fields"].get("errors", 0) + 1})
        continue

    if tokens > TOKENS_FOR_16K:
        gpt = EasyGPT(openai, "gpt-3.5-turbo-16k")
        logging.info("Using 16k model")

    prompt = CLASSIFY_PROMPT.replace("[article]", content)
    gpt.set_system_message(CLASSIFY_SYSTEM_MESSAGE)

    (answer, input_price, output_price) = gpt.ask(prompt)

    try:
        parsed_json = json.loads(answer)
        price = input_price + output_price
        
        if all(key in parsed_json for key in ["summary", "tags", "categories"]):
            calassification = {
                "summary": parsed_json["summary"],
                "tags": ", ".join(parsed_json["tags"]),
                "categories": ", ".join(parsed_json["categories"]),
                "tokens": tokens,
                "price": price
            }

            # check if we have countries in the response
            if "countries" in parsed_json:
                print("Countries: " + ", ".join(parsed_json["countries"]))
                if len(parsed_json["countries"]) > 0:
                    calassification["countries"] = ", ".join(parsed_json["countries"])
                    print("Adding countries: " + calassification["countries"])

            logging.info("JSON valid, saving to Airtable")
            table_news.update(article["id"], calassification, typecast=True)
        else:
            logging.error("JSON missing required fields, skipping")
            table_news.update(article["id"], {"errors": article["fields"].get("errors", 0) + 1})

    except json.JSONDecodeError:
        logging.error("Invalid JSON, skipping")
        table_news.update(article["id"], {"errors": article["fields"].get("errors", 0) + 1})
