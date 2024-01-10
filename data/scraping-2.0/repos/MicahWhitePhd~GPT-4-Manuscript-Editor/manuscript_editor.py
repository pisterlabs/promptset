import os
import glob
import sys
import time
import argparse
from manuscript_reader import DocxManuscriptReader, TxtManuscriptReader, CsvManuscriptReader
from manuscript_writer import DocxManuscriptWriter, TxtManuscriptWriter
from gpt_api_handler import GPTAPIHandler
from openai.error import RateLimitError  # Import RateLimitError

def read_prompt_file(prompt_file):
    with open(prompt_file, 'r') as file:
        return file.read()

def get_reader(file_path, start_text):
    if file_path.endswith(".docx"):
        return DocxManuscriptReader(file_path, start_text=start_text)
    elif file_path.endswith(".txt"):
        return TxtManuscriptReader(file_path, start_text=start_text)
    elif file_path.endswith(".csv"):
        return CsvManuscriptReader(file_path, start_text=start_text)
    else:
        raise ValueError(f"Unsupported file type for {file_path}")

def get_writer(file_path):
    if file_path.endswith(".docx"):
        return DocxManuscriptWriter(file_path)
    elif file_path.endswith(".txt"):
        return TxtManuscriptWriter(file_path)
    elif file_path.endswith(".csv"):
        return TxtManuscriptWriter(file_path)  # Assuming writing CSV as plain text for now
    else:
        raise ValueError(f"Unsupported file type for {file_path}")

def main(input_directory, output_directory, prompt_file, start_text=None):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Read the content of the prompt file
    with open(prompt_file, 'r') as file:
        prompt = file.read()

    # Get a list of all files in the input directory
    input_files = glob.glob(os.path.join(input_directory, "*"))

    # Create a file to store all chunks
    with open('temp_output.txt', 'w+') as temp_file:
        for input_file in input_files:
            # Determine the output file path
            base_name = os.path.basename(input_file)
            output_file = os.path.join(output_directory, base_name)

            # Determine the appropriate reader and writer based on the file extension
            if input_file.lower().endswith(".docx"):
                reader = DocxManuscriptReader(input_file, start_text=start_text)
                writer = DocxManuscriptWriter(output_file)
            elif input_file.lower().endswith(".txt"):
                reader = TxtManuscriptReader(input_file, start_text=start_text)
                writer = TxtManuscriptWriter(output_file)
            elif input_file.lower().endswith(".csv"):
                reader = CsvManuscriptReader(input_file, start_text=start_text)
                writer = TxtManuscriptWriter(output_file)
            else:
                print(f"Skipping file {input_file} with unrecognized extension.")
                continue

            gpt_api = GPTAPIHandler(prompt, api_key)

            chunks = list(reader.get_chunks())
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1} of {total_chunks} for file {input_file}")
                print(f"Sending text to GPT API: {chunk[:100]}...")

                # Modified part
                while True:
                    try:
                        edited_text = gpt_api.get_edited_text(chunk)
                        break  # Breaks the loop if the API call was successful
                    except Exception as e:
                        print(f"Error occurred: {str(e)}. Retrying after 60 seconds.")
                        time.sleep(60)  # Wait for 60 seconds before trying again



                print(f"Received edited text: {edited_text[:100]}...")

                # Write edited text to temp file
                temp_file.write(edited_text + "\n")
                temp_file.flush()  # Flush the buffer

                writer.add_paragraph(edited_text)

            # Save the document after all chunks have been processed
            writer.save() 

            print(f"Edited manuscript for file {input_file} saved to {output_file}")

        print(f"All edited texts have been recorded in the file: {temp_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manuscript Editor")
    parser.add_argument("input_directory", help="Input manuscript directory path")
    parser.add_argument("output_directory", help="Output edited manuscript directory path")
    parser.add_argument("prompt_file", help="Prompt file path")
    parser.add_argument("--start_text", help="Optional start text to process the input manuscript from", default=None)

    args = parser.parse_args()

    main(args.input_directory, args.output_directory, args.prompt_file, args.start_text)
