import json
import requests
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import csv
import openai

# Function to fetch metadata (title and description) of a webpage
def get_metadata(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else None
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title["content"]

        og_description = soup.find("meta", property="og:description")
        description = og_description["content"] if og_description else ""

        return f"{title}: {description}" if title else description
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for {url}: {e}")
        return None

# Function to fetch the first 3000 characters of text from a webpage
def fetch_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        text = ""
        for string in soup.stripped_strings:
            text += string
            if len(text) >= 3000:
                text = text[:3000]
                break

        # Remove non-ASCII characters
        text = "".join(char for char in text if ord(char) < 128)

        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching website text for {url}: {e}")
        return None

# Function to get a summary and best topic from GPT-3 for a given text
def gpt3_summarize_and_categorize(api_key, text, topics):
    openai.api_key = api_key

    summary_prompt = f"Please summarize the following text in 1000 characters or less, using a technical tone which does not describe or reference the source but simply presents the information:\n{text}\n\nSummary:"
    topic_prompt = f"Based on the following text, which topic best represents the content? The possible topics are:\n\n{text}\n\nTopics:\n{chr(10).join([f'- {topic}' for topic in topics])}\n\nThe best topic is:"

    response_summary = openai.Completion.create(
        engine="text-davinci-002",
        prompt=summary_prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response_topic = openai.Completion.create(
        engine="text-davinci-002",
        prompt=topic_prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    summary = response_summary.choices[0].text.strip() if response_summary.choices else None
    topic = response_topic.choices[0].text.strip() if response_topic.choices else None

    return summary, topic

# Function to find the parent element of a given element in an XML tree
def find_parent(element, root):
    for parent in root.iter("node"):
        for child in parent:
            if child == element:
                return parent
    return None

# Load API key
with open("api.key", "r") as key_file:
    api_key = key_file.read().strip()

# Load tasks from NoNonsenseNotes_Backup.json
with open("NoNonsenseNotes_Backup.json", "r") as json_file:
    data = json.load(json_file)
tasks = data["lists"][0]["tasks"]

# Parse the mindmap XML file
tree = ET.parse("fromNotes.mm")
root = tree.getroot()
main_node = root.find(".//node")

# Create or find the "IMPORT" node
import_node = main_node.find(".//node[@TEXT='IMPORT']")
if import_node is None:
    import_node = ET.SubElement(main_node, "node", {"TEXT": "IMPORT", "POSITION": "right"})

# Process tasks
for task in tasks:
    title = task["title"]

    # Extract URL from task title
    url = None
    for word in title.split():
        if urlparse(word).scheme:
            url = word
            break

    # Get metadata if URL is present
    if url:
        metadata = get_metadata(url)
        if metadata:
            title = metadata

    # Create a placeholder topic node for each task
    place_holder_topic = ET.SubElement(import_node, "node", {"TEXT": "place_holder_topic", "POSITION": "right"})

    # Nest the task node under the placeholder topic node
    new_node = ET.SubElement(place_holder_topic, "node", {"TEXT": title, "POSITION": "right"})
    
    if url:
        new_node.set("LINK", url)

# Read and parse the CSV file to create a list of topics
topics = []
with open("topics.txt", "r") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        for topic in row:
            topic = topic.strip()
            topics.append(topic)
            ET.SubElement(main_node, "node", {"TEXT": topic, "POSITION": "right"})

# Process nodes under "IMPORT"
import_nodes = []
for place_holder in import_node.findall("node"):
    import_nodes.extend(place_holder.findall("node"))

# Iterate through the stored list of nodes and fetch summaries and topics from GPT-3
for task_node in import_nodes:
    url = task_node.get("LINK")
    if url:
        print(f"Fetching website text for {url}")
        text = fetch_website_text(url)
        if text:
            print("Text sent to GPT-3:")
            print(text)
            print("Generating summary and category with GPT-3")
            summary, chosen_topic = gpt3_summarize_and_categorize(api_key, text, topics)
            if summary and chosen_topic:
                print(f"Summary: {summary}\nTopic: {chosen_topic}")

                # Create a new child node with the summary
                ET.SubElement(task_node, "node", {"TEXT": summary})

                # Overwrite the placeholder topic with the returned topic from OpenAI
                place_holder_topic = find_parent(task_node, import_node)
                place_holder_topic.set("TEXT", chosen_topic)

# Save the modified mindmap to a new file
tree.write("modified_mindmap.mm", encoding="utf-8", xml_declaration=True)

# Function to export topics and summaries as a LaTeX formatted file
def export_to_latex_file(tree, filename):
    root = tree.getroot()
    main_node = root.find(".//node")

    # Initialize the LaTeX formatted string
    latex_output = "\\chapter{Imported from phone}\n"

    # Iterate through topics and append the content to the LaTeX formatted string
    for topic_node in main_node.findall("node"):
        topic = topic_node.get("TEXT")
        if topic != "IMPORT":
            latex_output += f"\\subsubsection{{{topic}}}\n"

            # Iterate through the task_nodes within import_node
            for place_holder_topic in import_node.findall("node"):
                if place_holder_topic.get("TEXT") == topic:
                    for summary_node in place_holder_topic.findall("node/node"):
                        summary_text = summary_node.get("TEXT")
                        link = summary_node.get("LINK", "")

                        if link:
                            latex_output += f"(\\href{{{link}}}{{{link}}}) "

                        latex_output += f"{summary_text}\n\n"

    # Write the LaTeX formatted string to the specified file
    with open(filename, "w") as tex_file:
        tex_file.write(latex_output)

# Call the export_to_latex_file function to export the modified mindmap
export_to_latex_file(tree, "exportChapter.tex")