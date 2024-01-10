import argparse
import re
from dotenv import load_dotenv
import openai
import os
import sys


load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


DIR = "../mechanisms/"

MECHANISM_MAP = {
    "new": "AMCs.md",
    "legacy": "Municipal Bonds.md",
    "experimental": "Holographic Consensus.md"
}

HEADERS = ["Description", "Examples", "Further reading", "Acknowledgements"]

def generate_markdown(conversation_history, topic, tag, additional_text=None):
    # Update the conversation history with the new topic and additional text (if provided)
    conversation_history += f"User: Please create a markdown file about {topic}."
    conversation_history += f"It must have a title of {topic}, a tag of {tag}, and the following section headers {HEADERS}."
    conversation_history += f"The description should be 2-3 paragraphs only."
    if additional_text:
        conversation_history += f" Here's some additional context:\n{additional_text}\n"
    else:
        conversation_history += "\n"

    # Generate content using the ChatGPT model
    # TODO: review these settings
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=conversation_history,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.8,
            top_p=1
        )
    except Exception as e:
        print(f"Error generating markdown content: {e}")
        sys.exit(1)

    # Extract the generated markdown content
    markdown_content = response.choices[0].text.strip()
    conversation_history += f"Assistant: {markdown_content}\n"

    return markdown_content, conversation_history


def trim_markdown(markdown_content):
    # Split the markdown into lines
    lines = markdown_content.split("\n")

    # Find the index of the first level-1 header
    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            header_index = i
            break

    if header_index is None:
        # No header found, return the original markdown
        return markdown_content
    else:
        # Trim any text before the header
        lines = lines[header_index:]
        lines[0] = re.sub(r"^[^#]*", "", lines[0])
        return "\n".join(lines)


def save_markdown_to_file(filename, content):
    content = trim_markdown(content)
    try:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Generated markdown saved as '{filename}'")
    except Exception as e:
        print(f"Error saving generated markdown: {e}")
        sys.exit(1)

def read_additional_text(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading additional text file: {e}")
        sys.exit(1)


def check_markdown_conformance(markdown_content):

    # Extract the section headers from the markdown
    section_headers = re.findall(r"#{2,3}\s+([^\n]+)", markdown_content, flags=re.IGNORECASE)

    # Check that the section headers match the expected headers
    for section in HEADERS:
        if section not in section_headers:
            return False

    return True


def conversation(args):
    # Initialize with sample markdown
    mechanism_file = MECHANISM_MAP.get(args.mechanism_type)
    if mechanism_file:
        sample_markdown = read_additional_text(os.path.join(DIR, mechanism_file))
        conversation_history = f"Assistant: Here's a sample markdown provided by the user:\n\n{sample_markdown}\n"
    else:
        print(f"Error: Invalid mechanism type '{args.mechanism_type}'.")
        sys.exit(1)

    # Read additional text from the local markdown file (if provided)
    additional_text = None
    if args.additional_text_file:
        additional_text = read_additional_text(args.additional_text_file)

    while True:
        # Generate markdown content
        generated_markdown, conversation_history = generate_markdown(
            conversation_history, 
            args.topic, 
            args.mechanism_type, 
            additional_text)

        # Save the generated markdown to a file
        save_markdown_to_file(os.path.join(DIR, args.output_filename), generated_markdown)

        # Check that the generated markdown conforms to the template
        if check_markdown_conformance(generated_markdown):
            break
        else:
            print("The generated markdown does not conform to the template. Starting conversation again...")

# TODO: automate this workflow from CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate markdown using OpenAI GPT-3.')
    parser.add_argument('topic', type=str, help='the topic for the generated markdown')
    parser.add_argument('mechanism_type', type=str, choices=["new", "legacy", "experimental"], default="new", help='the type')
    parser.add_argument('output_filename', type=str, help='the filename for the generated markdown file')
    parser.add_argument('--additional-text-file', type=str, help='the path to a file containing additional text for the generated markdown')
    args = parser.parse_args()

    conversation(args)
