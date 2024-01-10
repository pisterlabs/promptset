import tiktoken
import re
import argparse
import os
from openai import OpenAI

# gpt-3.5-turbo , gpt-4-1106-preview
# Your OpenAI API key
OPENAI_API_KEY = "YOUR_KEY"
OPENAI_TEXT_MODEL = "gpt-4-1106-preview"
OPENAI_MAX_SUMM_TOKENS = 4096
OPENAI_MAX_TOKEN_PER_SECTION = 128000

# Initialize the OpenAI API client
client = OpenAI(api_key=OPENAI_API_KEY)
encoding = tiktoken.encoding_for_model(OPENAI_TEXT_MODEL)


def count_tokens(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_text_into_sections(text, max_tokens):
    sections = []
    current_section = ""
    current_token_count = 0

    for line in text.split('\n'):
        tokens = count_tokens(line)

        if current_token_count + tokens <= max_tokens:
            current_section += line + '\n'
            current_token_count += tokens
        else:
            sections.append(current_section.strip())
            current_section = line + '\n'
            current_token_count = tokens

    if current_section:
        sections.append(current_section.strip())

    return sections


def summarize_text(text, prompt=None):
    if prompt is None:
        prompt = """
        Supplied are transcriptions of radio traffic.
        You are writing a daily summary of high profile incidents for the Sheriff.
        List notable incidents that occurred in the transcriptions. 
        The file names provide more context to each message.
        They include the date and time of the transcription, the talkgroup ID, and
        the radio ID of who initiated the transmission.
        If there are names, callsigns, or addresses relevant to the highlight ensure to include them in the reported event so we can search for them later.
        Ensure the final summary is detailed and includes identifying details for each incident.
        List each incident in this format:
        - Desc: description of the incident
        - Address: any addresses mentioned in the incident
        - Names: any names mentioned in the incident
        - Filename: filename(s) of the transcription(s) utilized for summary
        """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt + text,
            }
        ],
        model=OPENAI_TEXT_MODEL)
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Summarize text files")
    parser.add_argument("-d", "--date", required=True,
                        help="Date in YYYYMMDD format")
    parser.add_argument("-f", "--folder", required=True,
                        help="Folder path containing text files")
    args = parser.parse_args()

    date = args.date
    folder_path = args.folder
    print(f"Date: {date}")
    print(f"Folder path: {folder_path}")

    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return

    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") and re.match(rf"{date}", file):
                txt_files.append(os.path.join(root, file))

    if not txt_files:
        print(f"No .txt files found in the folder for the date {date}.")
        return

    all_text = ""
    for txt_file in txt_files:
        with open(os.path.join(folder_path, txt_file), 'r') as file:
            file_content = file.read()
            all_text += f"Filename: {txt_file}\n{file_content}\n\n"

    total_tokens = count_tokens(all_text)
    print(f"Total tokens in the text: {total_tokens}")

    sections = split_text_into_sections(all_text, OPENAI_MAX_TOKEN_PER_SECTION)

    summaries = []
    for i, section in enumerate(sections):
        prompt = """
        Supplied are transcriptions of radio traffic.
        You are writing a daily summary of high profile incidents for the Sheriff.
        List notable incidents that occurred in the transcriptions. 
        The file names provide more context to each message.
        They include the date and time of the transcription, the talkgroup ID, and
        the radio ID of who initiated the transmission.
        If there are names, callsigns, or addresses relevant to the highlight ensure to include them in the reported event so we can search for them later.
        Ensure the final summary is detailed and includes identifying details for each incident.
        List each incident in this format:
        - Desc: description of the incident
        - Address: any addresses mentioned in the incident
        - Names: any names mentioned in the incident
        - Filename: filename(s) of the transcription(s) utilized for summary
        """
        summary = summarize_text(section, prompt=prompt)
        summaries.append(summary)

    for i, summary in enumerate(summaries):
        print(f"Summary for section {i + 1}:\n{summary}")


if __name__ == "__main__":
    main()
