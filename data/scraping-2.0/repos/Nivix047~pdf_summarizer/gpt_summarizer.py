import nltk
import PyPDF2
import openai
import os
from dotenv import load_dotenv
from collections import deque
from sacremoses import MosesDetokenizer

# Check if the 'punkt' tokenizer models are available, download them if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer models...")
    nltk.download('punkt')

# Function to extract text from PDF


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page_obj = pdf_reader.pages[page_num]
                text += page_obj.extract_text()
        if not text:
            print("Warning: No text was extracted from the PDF.")
        return text
    except Exception as e:
        print(
            f"An error occurred while extracting text from the PDF: {str(e)}")
        return None

# Function to summarize text using OpenAI GPT


def gpt_summarize(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a meeting assistant who summarizes the meeting minutes."},
                {"role": "user", "content": f"{text}\n\nSummarize:"},
            ]
        )
        return response
    except Exception as e:
        print(f"An error occurred during summarization: {str(e)}")
        return None

# Function to recursively summarize text


def recursively_summarize(text, section_size=3000, overlap=150):
    detokenizer = MosesDetokenizer()
    tokens = nltk.word_tokenize(text)
    summaries = []

    # Create overlapping sections
    sections = deque()
    start = 0
    while start < len(tokens):
        sections.append(tokens[start:start + section_size])
        start += section_size - overlap

    # Summarize each section and append to the summaries list
    while len(sections) > 1:
        section = sections.popleft()
        summary = gpt_summarize(detokenizer.detokenize(section))
        if summary is not None:
            summaries.append(summary)
        else:
            print("Error: Summarization failed.")
            return None

        # Break the summarized sections into smaller sections again
        if len(summaries) > 1:
            sections.append(nltk.word_tokenize(summaries.pop()) +
                            nltk.word_tokenize(summaries.pop()))
        else:
            return summaries[0]


if __name__ == "__main__":
    # Set OpenAI API key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Specify PDF path
    pdf_path = "/Users/nivix047/Desktop/Meeting_minutes.pdf"
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        exit(1)

    # Alternatively, read text from a file
    # with open('minutes.txt', 'r') as f:
    #     text = f.read()

    # Summarize text
    response = recursively_summarize(text)
    if response is None:
        exit(1)

    # Print summarized text
    if 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
        print(response['choices'][0]['message']['content'])
    else:
        print("Error: The response from the OpenAI API was not in the expected format.")
        exit(1)
