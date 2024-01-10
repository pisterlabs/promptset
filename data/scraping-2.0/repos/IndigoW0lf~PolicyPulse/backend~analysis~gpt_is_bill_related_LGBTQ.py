from config import Config
from dotenv import load_dotenv
from lxml import etree
import logging
import openai
import os
import shutil

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the logger
logging.basicConfig(
    filename='backend/database/logs/gpt_bill_relevancy.log',
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


def is_bill_related_to_lgbtq(text):
    """Determine if a given bill text is related to LGBTQ+ rights issues using GPT-3.5-turbo."""
    logger.debug("Sending request to GPT to analyze bill relevance.")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": ("Given the context that LGBTQ+ rights issues pertain to topics like anti-discrimination, "
                            "marriage equality, transgender rights, gender identity, sexual orientation, and other related civil rights topics, "
                            "please analyze the following bill content and determine its relevance. "
                            "If the bill is related to LGBTQ+ rights issues, respond with 'LGBTQ_RELEVANT'. "
                            "If it's not related, respond with 'NOT_RELEVANT'.")
            },
            {"role": "system", "content": text}
        ]
    )
    gpt_output = response.choices[0].message.content
    logger.debug(f"GPT Output: {gpt_output}")
    return gpt_output


def process_xml_files(directory):
    # logging.info(f"Processing XML files in directory: {directory}")

    xml_files = [os.path.join(root, f) for root, _, files in os.walk(
        directory) for f in files if f.endswith('.xml')]

    relevant_bills = 0
    non_relevant_bills = 0

    for xml_file in xml_files:
        # logger.debug(f"Processing XML file: {xml_file}")
        try:
            tree = etree.parse(xml_file)

            # Extracting the title, summary, and amended bill title text
            title_element = tree.find(
                ".//titles/item[titleType='Official Title as Introduced']/title")
            summary_element = tree.find(".//summaries/summary/text")
            amended_title_element = tree.find(".//amendedBill/title")

            # Getting the text content from each element, if the element was found
            title_text = title_element.text if title_element is not None else ""
            summary_text = summary_element.text if summary_element is not None else ""
            amended_title_text = amended_title_element.text if amended_title_element is not None else ""

            combined_text = title_text + summary_text + amended_title_text

            relevance = is_bill_related_to_lgbtq(combined_text)

            # Log GPT's response for each bill
            logging.info(
                f"\n Summary text for bill '{xml_file}': {summary_text} \n GPT's response for bill '{xml_file}': {relevance}")

            # Modify the condition to check for the specific keyword
            if "LGBTQ_RELEVANT" in relevance:
                relevant_bills += 1

                # Move the relevant file to the 'prelim_related_xml_files' folder
                destination_folder = "backend/data/LGBTQ+_related_xml_files"
                shutil.move(xml_file, os.path.join(
                    destination_folder, os.path.basename(xml_file)))

            else:
                non_relevant_bills += 1

                # Move the non-relevant file to the 'prelim_non_related_xml_files' folder
                destination_folder = "backend/data/prelim_non_related_xml_files"
                shutil.move(xml_file, os.path.join(
                    destination_folder, os.path.basename(xml_file)))

        except Exception as e:
            logger.error(f"Error processing file {xml_file}", exc_info=True)

    logger.info(
        f"Finished processing XML files. Total relevant bills: {relevant_bills}. Total non-relevant bills: {non_relevant_bills}.")


# Specify the directory containing XML files
if __name__ == "__main__":
    process_xml_files('backend/data/prelim_related_xml_files')
