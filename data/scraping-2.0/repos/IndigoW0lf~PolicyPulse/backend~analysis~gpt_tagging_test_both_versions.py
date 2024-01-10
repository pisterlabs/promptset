from config import Config
from dotenv import load_dotenv
from lxml import etree
import logging
import openai
import os
from backend.data.tag_list import tags

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the logger
logging.basicConfig(
    filename='backend/database/logs/test_both_versions_tagging.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
)

# Create logger instance
logger = logging.getLogger(__name__)

# Add a console handler to the logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] - %(message)s'))
logger.addHandler(console_handler)


def get_tags_with_chat_model(bill_content):
    """Get tags for a bill using the gpt-3.5-turbo-16k model."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
                "content": f"Given the following bill content: '{bill_content}', and considering the list of tags: {', '.join(tags)}, suggest the most relevant tags for this bill. Note: Only answer with the tags. You can suggest more than one tag, separated by commas."}
        ]
    )
    suggested_tags = response.choices[0].message['content'].strip().split(", ")
    return suggested_tags


def get_tags_with_completion_model(bill_content):
    """Get tags for a bill using the default completion model."""
    prompt = f"Given the following bill content: '{bill_content}', and considering the list of tags: {', '.join(tags)}, suggest the most relevant tags for this bill. Note: Only answer with the tags. You can suggest more than one tag, separated by commas."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )
    suggested_tags = response.choices[0].text.strip().split(", ")
    return suggested_tags


def process_xml_files(directory):
    xml_files = [os.path.join(root, f) for root, _, files in os.walk(
        directory) for f in files if f.endswith('.xml')]

    for xml_file in xml_files:
        try:
            tree = etree.parse(xml_file)

            # Extracting the title, summary, and amended bill title text
            title_element = tree.find(
                ".//titles/item[titleType='Display Title']/title")
            summary_element = tree.find(".//summaries/summary/text")
            amended_title_element = tree.find(".//amendedBill/title")

            # Getting the text content from each element, if the element was found
            title_text = title_element.text if title_element is not None else ""
            summary_text = summary_element.text if summary_element is not None else ""
            amended_title_text = amended_title_element.text if amended_title_element is not None else ""

            combined_text = title_text + summary_text + amended_title_text

            tags_from_chat_model = get_tags_with_chat_model(combined_text)
            tags_from_completion_model = get_tags_with_completion_model(
                combined_text)

            logger.info(f"Bill '{xml_file}':")
            logger.info(f"Tags from Chat Model: {tags_from_chat_model}")
            logger.info(
                f"Tags from Completion Model: {tags_from_completion_model}")

        except Exception as e:
            logger.error(f"Error processing file {xml_file}", exc_info=True)


# Specify the directory containing XML files
if __name__ == "__main__":
    process_xml_files('backend/data/test_xml_files')
