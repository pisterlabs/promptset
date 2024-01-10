from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import re
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime, timedelta

from bs4 import BeautifulSoup

load_dotenv()


def replace_smart_quotes(text):
    """
    This function replaces smart quotes with standard quotes in a given text.
    """
    smart_quotes_dict = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201A": "'",  # Single low-9 quotation mark
        "\u201B": "'",  # Single high-reversed-9 quotation mark
        "\u201C": '"',  # Left double quotation mark
        "\u201D": '"',  # Right double quotation mark
        "\u201E": '"',  # Double low-9 quotation mark
        "\u201F": '"',  # Double high-reversed-9 quotation mark
        "\u2032": "'",  # Prime (minutes, feet)
        "\u2033": '"',  # Double Prime (seconds, inches)
    }
    for k, v in smart_quotes_dict.items():
        text = text.replace(k, v)
    return text


def clean_invalid_chars(text, encoding="utf-8"):
    """
    This function replaces characters that cause issues in encoding/decoding with a replacement character.
    """
    byte_string = text.encode(encoding, "replace")
    clean_text = byte_string.decode(encoding, "replace")
    return clean_text


chat = ChatAnthropic()

srt_dir = "latentspacepodcast/srt/"
chap_dir = "latentspacepodcast/chapters/"

if not os.path.exists(chap_dir):
    os.makedirs(chap_dir)


# Extract number from filename
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


def convert_time_str_to_seconds(time_str):
    formats = ["%M:%S", "%H:%M:%S"]

    if "," in time_str:
        time_str = time_str.split(",")[0]  # Remove milliseconds
    if time_str.startswith("00:"):
        time_str = time_str[3:]  # Trim leading zero hours

    for time_format in formats:
        try:
            time_object = datetime.strptime(time_str, time_format)
            total_seconds = (
                time_object.hour * 3600
                + time_object.minute * 60
                + time_object.second
                + time_object.microsecond / 1e6
            )
            return int(total_seconds)
        except ValueError:
            pass  # Incorrect format, let's try the next one

    # Let's manually handle time string where hour > 23
    try:
        parts = time_str.split(":")
        total_seconds = 0
        if len(parts) == 3:
            total_seconds += int(parts[0]) * 3600  # hour to seconds
            total_seconds += int(parts[1]) * 60  # minute to seconds
            total_seconds += int(parts[2])  # seconds
        elif len(parts) == 2:
            total_seconds += int(parts[0]) * 60  # minute to seconds
            total_seconds += int(parts[1])  # seconds
        return total_seconds
    except ValueError:
        pass

    raise ValueError(f"Time '{time_str}' does not match any expected format {formats}")


# Get list of files in directory
files = os.listdir(srt_dir)

# Sort files by number in filename, from largest to smallest
sorted_files = sorted(files, key=extract_number, reverse=True)

for filename in sorted_files[0:1]:
    with open(srt_dir + filename, "r") as f:
        transcript = f.read().strip()

    if transcript == "":
        continue

    finalXML = ""
    chat_prompt_arr = []
    for i in range(20):
        if i == 0:
            humanTemplate = '<transcript> {transcript} </transcript> \n\n Please break this podcast into 6 sections based on topics with timestamps. Now act as a XML code outputter. Please, Do not add any additional context or introduction in your response, make sure your entire response is parseable by xml. Also do not use single quotes: "<episode> <segment><title>Intro</title><startTime>0:00</startTime><endTime>5:29</endTime></segment> </episode>"'
        else:
            humanTemplate = "continue"
        humanMessagePrompt = HumanMessagePromptTemplate.from_template(humanTemplate)

        aiMessagePrompt = AIMessagePromptTemplate.from_template(finalXML)

        chat_prompt_arr.append(humanMessagePrompt)
        chat_prompt_arr.append(aiMessagePrompt)

        chat_prompt = ChatPromptTemplate.from_messages(chat_prompt_arr)
        result = chat(chat_prompt.format_prompt(transcript=transcript).to_messages())
        finalXML += result.content

        # Regular expression to match XML in text
        xml_regex = r"<episode>.*?</episode>"
        match = re.search(xml_regex, finalXML, re.DOTALL)

        xml_regex = r"<transcript>.*?</transcript>"
        match2 = re.search(xml_regex, finalXML, re.DOTALL)

        if match or match2:
            break

    # Extract matched XML
    xml_text = match.group(0) if match else match2.group(0)
    # xml_text_clean = replace_smart_quotes(xml_text)
    # xml_text_clean = clean_invalid_chars(xml_text_clean)

    soup = BeautifulSoup(xml_text, "xml")
    # Convert the BeautifulSoup object back to a string
    xml_text = str(soup)
    # Parse XML
    root = ET.fromstring(xml_text)

    chapters = []

    for segment in root.iter("segment"):
        title = segment.find("title").text
        start_time = convert_time_str_to_seconds(
            segment.find("startTime").text
        )  # convert to seconds

        # Create a dictionary for this segment
        chapter = {
            "title": title,
            "startTime": start_time,
        }

        # Add the dictionary to the list
        chapters.append(chapter)

    # Create a dictionary with the chapters and version
    data = {
        "chapters": chapters,
        "version": "1.0.0",
    }

    # Write the dictionary to a new JSON file
    with open(chap_dir + filename + "_chapters.json", "w") as f:
        json.dump(data, f)
