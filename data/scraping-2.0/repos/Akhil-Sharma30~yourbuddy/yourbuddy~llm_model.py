
from llama_index import VectorStoreIndex, download_loader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path
from github import Github
import os
import shutil
import openai
import pypandoc

from pathlib import Path
from llama_index import download_loader

# OPENAI key
openai.api_key = os.environ.get("OPENAPI_API_KEY")

"""# Reading the Files for LLM Model"""


class Llm_Training:

    def get_file_input():
        # Ask the user for their choice (link or upload)
        print("How would you like to provide the file?")
        print("1. Link")
        print("2. Upload")

        choice = input("Enter your choice (1 or 2): ")

        # Validate the user's choice
        if choice not in ['1', '2']:
            print("Invalid choice. Please enter '1' for link or '2' for upload.")
            return None

        if choice == '1':
            # If the user chose 'link', ask for the link and store it in a variable
            file_link = input("Enter the file link: ")
            return file_link
        else:
            # If the user chose 'upload', you can implement the file upload logic here
            # and return the uploaded file's path or content.
            # Replace with your upload logic
            uploaded_file = input("Upload the file: ")
            return uploaded_file

    # Converts a file to markdown format.
    def convert_to_markdown(input_file, output_file):

        # Args:
        # input_file (str): The path to the input file.
        # output_file (str): The path to the output markdown file.

        # Returns:
        # bool: True if conversion is successful, False otherwise.

        try:
            # Convert the input file to markdown format
            pypandoc.convert_file(input_file, 'markdown',
                                  outputfile=output_file)
            return True
        except Exception as e:
            print(f"Conversion error: {str(e)}")
            return False

    def chatbot_choice(user_input):
        # Ask the user for their choice (personal data or general chatbot)
        print("Choose an option:")
        print("1. Train on personal data")
        print("2. Use a general chatbot (like OpenAI)")

        choice = input("Enter your choice (1 or 2): ")

        # Validate the user's choice
        if choice not in ['1', '2']:
            print(
                "Invalid choice. Please enter '1' for personal data or '2' for a general chatbot.")
            return

        if choice == '1':
            # User selected training on personal data
            print("Training on personal data...")
            response_data = chatbot_function(user_input)

        else:
            # User selected using a general chatbot
            print("Using a general chatbot like OpenAI...")
            response_data = Generative_response(user_input)

    def chatbot_function(query):
        # Set up the OpenAI API key
        openai.api_key = os.environ.get("OPENAPI_API_KEY")

        # Load the MarkdownReader
        MarkdownReader = download_loader("MarkdownReader")
        loader = MarkdownReader()

        # Load user data from the "./output.md" file
        user_data_reader = loader.load_data(file="./output.md")

        # Vector Embedding
        index = VectorStoreIndex.from_documents(user_data_reader)

        # Create a query engine
        query_engine = index.as_query_engine()

        # Query the engine with the user's query
        response = query_engine.query(query)
        return response

    def Generative_response(prompt):
        # Generate a response using the GPT-3.5-turbo model
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": ""
                }
            ],
            temperature=1,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return gpt_response.choices[0].message["content"]


# Example usage:
if __name__ == "__main__":
    output_file = "output.md"  # Replace with your desired output file path
    input_file = Llm_Training.get_file_input()  # Replace with your input file path
    if Llm_Training.convert_to_markdown(input_file, output_file):
        print(
            f"File '{input_file}' converted to markdown successfully and saved as '{output_file}'.")
    else:
        print(f"Failed to convert '{input_file}' to markdown.")
