from typing import List
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import re


def detect_language(code: str):
    try:
        lexer = guess_lexer(code)
        return lexer.name
    except ClassNotFound:
        return "Unknown language"

def huggingface_tokenizer_length(text: str, tokenizer) -> int:
    tokenized_text = tokenizer(text, truncation=True, max_length=512)["input_ids"]
    return len(tokenized_text)

# Initialize the tokenizer once
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the chunk size and overlap
chunk_size = 2000
chunk_overlap = 5

class LanguageAwareTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_separators = {
            "Python": ["\n\n", "\n", " ", ""],
            "Rust": ["\nfn ", "\nconst ", "\nlet ", "\nif ", "\nwhile ", "\nfor ", "\nloop ", "\nmatch ", "\nconst ", "\n\n", "\n", " ", ""],
            "JS": [
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            "SHELL": [
            "\nfunction ",
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\ncase ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "MARKDOWN": [
                "\n#{1,6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            "default": ["\n\n", "\n", " ", ""]
        }

        def split_text(self, text: str) -> List[str]:
            chunks = []
            current_chunk = ""

            # Define a regex pattern that finds code blocks and the text immediately before and after them
            code_block_pattern = r"(.*?\n\n)?(```.*?```)(\n\n.*?)?"
            pattern = re.compile(code_block_pattern, re.DOTALL)

            # Initialize the start index of the next search
            start_idx = 0

            # Search for the next code block along with its surrounding text
            while True:
                match = pattern.search(text, start_idx)
                if not match:
                    break
                
                # Extract text segments before, the code block itself, and after
                before, code_block, after = match.groups()
                before = before or ""  # Ensure 'before' is a string if it's None
                after = after or ""    # Ensure 'after' is a string if it's None

                # Accumulate 'before' text and check for chunk splitting
                if before:
                    current_chunk = self._accumulate_and_maybe_split(current_chunk + before, chunks)

                # Always keep the code block in the same chunk
                current_chunk += code_block
                current_chunk = self._accumulate_and_maybe_split(current_chunk, chunks)

                # Accumulate 'after' text and check for chunk splitting
                if after:
                    current_chunk = self._accumulate_and_maybe_split(current_chunk + after, chunks)
                
                # Update the start index for the next search
                start_idx = match.end()

            # Append any remaining text after the last code block
            remaining_text = text[start_idx:]
            if remaining_text.strip():
                current_chunk = self._accumulate_and_maybe_split(current_chunk + remaining_text, chunks)
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        def _accumulate_and_maybe_split(self, text, chunks):
            if huggingface_tokenizer_length(text, tokenizer) > self._chunk_size:
                language = detect_language(text)
                self.separators = self.language_separators.get(language, self.language_separators["default"])
                chunk_splits = super().split_text(text)

                # Add all but the last piece to chunks
                chunks.extend(chunk_splits[:-1])

                # Start the next chunk with the last piece
                return chunk_splits[-1]
            return text

# ... (rest of your code)

# Create the text splitter
splitter = LanguageAwareTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap,
    length_function=lambda text: huggingface_tokenizer_length(text, tokenizer),
)

def append_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)

# Function to read text from a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Define the path to the text file
file_path = "/Users/jichen/Projects/rust-bert/splitter/book.txt"
file_path = "/Users/jichen/Projects/rust-bert/splitter/k8s.md"

# Read the text from the file
text = read_file(file_path)

# Split the text
splits = splitter.split_text(text)

# Print the splits
for idx, split in enumerate(splits, start=1):
    append_to_file("splits-detect-truncate-380-k8s.txt", f"Split {idx}:\n{split}\n\n")
