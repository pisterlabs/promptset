import configparser
import json

import ebooklib
import textract
from bs4 import BeautifulSoup
from ebooklib import epub
from openai import OpenAI

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize the OpenAI client
client = OpenAI(
    api_key=config['DEFAULT']['OPENAI_API_KEY']
)

DEFAULT_EXTRACTION = {"characters": [{"name": "Charlotte",
                                      "description": "A determined and passionate writer who misses the connect that she used to have with her audience. She is the creator of a story about a lonely girl with magical powers and a loving prince. She is a bit self-conscious and desires for someone to love the story she has created. Dresses up to look like a serious author. Anxious and biting on her thumbnail when contemplating inviting someone to read her story."},
                                     {"name": "Roget",
                                      "description": "A paid reader with a polite demeanor. He is tall, handsome with large lips, thin nose, and a rakish hairstyle. His attire is often semi-formal with khakis and a cornflower blue buttoned-down shirt. He appears to be wealthy, based on the narrative cues given in the text. He is also described as being very expressive and his enjoyment of the story is visible in his non-verbal cues. Has genuine interest in the work he is reviewing."}],
                      "locations": [{"name": "Coffee shop",
                                     "description": "A quiet place where Roget and Charlotte meet for him to read her book. They occupy a booth at the back, away from distractions."},
                                    {"name": "Internet Writing platforms - WattPad, AO3, and RoyalRoad",
                                     "description": "Once thriving online platforms where diverse narratives created by human writers used to enthrall numerous readers. However, following the advent of a deep learning model that could produce stories better than a human, the platforms lost their audience and authenticity."}],
                      "props": [{"name": "Laptop",
                                 "description": "A piece of technology Roget uses to read Charlotte\"s novel. It is in dark mode and also used for maintaining notes related to the read."},
                                {"name": "Flash Drive",
                                 "description": "Charlotte carries the novel she has written on this. It is passed to Roget to read."}],
                      "artstyle": "Realistic, given the emotional depth and complexity of human interactions within a modern, technologically advanced setting."}


def split_text_into_sections(text: str, max_chars=20000) -> list:
    """
    Splits the given text into sections based on the max_chars limit, ensuring
    no paragraphs are split in half.
    """
    paragraphs = text.split("\n")
    sections = []
    current_section = ""

    for paragraph in paragraphs:
        if len(current_section) + len(paragraph) < max_chars:
            current_section += paragraph + "\n"
        else:
            sections.append(current_section.strip())
            current_section = paragraph + "\n"
    if current_section:
        sections.append(current_section.strip())

    return sections


def extract_descriptions_from_section(section: str, local_mode: bool = False) -> dict:
    """
    Uses GPT-4 to extract character, prop, and location descriptions from the provided section of text.
    """
    # System prompt from the saved file
    with open("./prompts/extract_descriptions_system_prompt.txt", "r") as f:
        system_prompt = f.read()

    if not local_mode:
        # Call to GPT-4
        print("Calling GPT-4 for description extraction...")
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": section}
            ]
        )

        response_content = json.loads(completion.choices[0].message.content)
        print("Response from GPT-4 for description extraction:")
        print(response_content)
    else:
        response_content = DEFAULT_EXTRACTION
    return response_content


def enhance_descriptions(data: list[dict], local_mode: bool = False) -> dict:
    """
    Enhances sparse or ambiguous descriptions.
    """
    # Construct the user prompt
    user_prompt = str(data)

    # System prompt from the saved file
    with open("./prompts/consolidate_descriptions_system_prompt.txt", "r") as f:
        system_prompt = f.read()

    if not local_mode:
        # Call to GPT-4
        print("Calling GPT-4 for description enhancement...")
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        response_content = json.loads(completion.choices[0].message.content)
        print("Response from GPT-4 for description enhancement:")
        print(response_content)
    else:
        return DEFAULT_EXTRACTION
    return response_content


def extract_chapters_from_epub(filepath: str) -> list:
    """
    Extracts chapters from the given EPUB file.
    """
    book = epub.read_epub(filepath)
    chapters = []

    for item in book.items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(item.content, 'html.parser')

            # Replace <i> and <b> tags with spaces to prevent words from merging
            for tag in soup.find_all(['i', 'b']):
                tag.replace_with(f" {tag.get_text()} ")

            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            chapter_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])

            # Join broken sentences
            lines = chapter_text.split('\n')
            for i in range(len(lines) - 1):
                if not lines[i].endswith(('.', ',', '!', '?', ';', ':', '-', '”')):
                    lines[i + 1] = lines[i] + ' ' + lines[i + 1]
                    lines[i] = ''
            chapter_text = '\n'.join(filter(None, lines))

            # Only include chapters that exceed 700 words
            if chapter_text and len(chapter_text.split()) > 700:
                chapters.append(chapter_text)

    return chapters


def extract_content_as_chapters(filepath: str) -> list:
    """
    Extracts content from the file and returns it as a list of chapters.
    For EPUBs, it returns the actual chapters. For other formats, it returns a list with a single entry.
    """
    if filepath.endswith('.epub'):
        return extract_chapters_from_epub(filepath)
    else:
        return [textract.process(filepath).decode('utf-8')]
