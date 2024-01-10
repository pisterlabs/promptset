import argparse
import os
import textwrap
import uuid

from directory import remove_all_items
from tqdm import tqdm
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.schema import Document


class Manipulator:
    def __init__(self, args: argparse.Namespace = None) -> None:
        self.folder_path = "docs"
        self.args = args
        self.create_folder()

    def create_folder(self):
        try:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
                print("The 'docs' folder has been created successfully.")
            else:
                print("The 'docs' folder already exists.")
        except OSError as e:
            print(f"An error occurred while creating the 'docs' folder: {str(e)}")

    def read_and_split(self, documents: list[Document], max_words_per_file: int, wrap: bool = False):
        remove_all_items(self.folder_path)

        # Combine all page contents from the documents into a single string
        words = " ".join(document.page_content for document in documents).split()

        # Calculate the number of files needed to accommodate the words
        num_files = (len(words) + max_words_per_file - 1) // max_words_per_file

        # Iterate over the number of files to write the split text content
        for i in tqdm(range(num_files), desc="Writing document"):
            # Generate a unique file name using UUID
            txt_file = os.path.join(self.folder_path, f"{uuid.uuid4()}.txt")

            with open(txt_file, "w", encoding="utf-8") as file:
                # Determine the start and end indices of the words for the current file
                start_index = i * max_words_per_file
                end_index = min((i + 1) * max_words_per_file, len(words))

                # Extract the relevant words for the current file
                text_part = " ".join(words[start_index:end_index])

                if wrap:
                    # Wrap the text into paragraphs with a width of 70 characters
                    wrapped_text = textwrap.fill(text_part, width=70)
                    file.write(wrapped_text.lower())
                else:
                    # Write the text without wrapping
                    file.write(text_part.lower())

    def process_pdf(self, file_path: str):
        pdf_reader = UnstructuredPDFLoader(file_path)
        documents = pdf_reader.load_and_split()
        self.read_and_split(documents, self.args.max_word, self.args.wrap)

    def process_word(self, file_path: str):
        word_reader = UnstructuredWordDocumentLoader(file_path)
        documents = word_reader.load_and_split()
        self.read_and_split(documents, self.args.max_word, self.args.wrap)

    def run(self):
        if not 100 <= self.args.max_word <= 700:
            print("Maximum word limit should be between 100 and 700 (inclusive).")
            exit(0)

        if self.args.pdf and self.args.word:
            print("Error: Cannot use both --pdf and --word arguments simultaneously.")
        elif self.args.pdf:
            self.process_file(self.args.pdf, "PDF")
        elif self.args.word:
            self.process_file(self.args.word, "Word")
        elif self.args.clean:
            remove_all_items(self.folder_path)
        else:
            print("No valid input provided.")

    def process_file(self, file_location, file_type):
        if not file_location:
            print(f"Error: Please provide the location of the {file_type} file using the --{file_type.lower()}-location argument.")
        else:
            try:
                if file_type == "PDF":
                    self.process_pdf(file_location)
                elif file_type == "Word":
                    self.process_word(file_location)
            except ValueError as e:
                print(f"Error: {str(e)}")


def check_file_type(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    supported_extensions = {".pdf", ".docx", ".doc"}

    if file_extension not in supported_extensions:
        raise ValueError("Unsupported file type. Please provide a PDF or Word document.")

    if file_extension == ".pdf":
        return "PDF"
    elif file_extension in {".docx", ".doc"}:
        return "Word"



def parse_arguments():
    parser = argparse.ArgumentParser()

    file_group = parser.add_argument_group("File options")
    file_group.add_argument("--pdf", type=str, help="Path of the PDF file to be read")
    file_group.add_argument("--word", type=str, help="Path of the Word file to be read")
    file_group.add_argument("--max-word", type=int, default=400, help="Maximum word limit per file (default: 400, min:100, max: 700)")

    process_group = parser.add_argument_group("Processing options")
    process_group.add_argument("--clean", action="store_true", help="Remove the 'docs' folder before processing the file")
    process_group.add_argument("--wrap", action="store_true", help="Wrap the text into paragraphs with a width of 70 characters")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    manipulator = Manipulator(args)
    manipulator.run()
