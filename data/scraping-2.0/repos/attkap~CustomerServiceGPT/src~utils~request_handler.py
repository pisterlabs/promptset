import logging
import os
from typing import Dict

from .customer_request import CustomerRequest
from .data_processor import DataProcessor
from .openai_api import OpenAI_API


class RequestHandler:
    def __init__(self, input_dir: str, output_dir: str) -> None:
        """
        Initialize a RequestHandler.

        :param input_dir: Directory containing input files.
        :param output_dir: Directory where to save output files.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create instances of utility classes
        self.openai_api = OpenAI_API()
        self.data_processor = DataProcessor(input_dir, output_dir)

        # Get a logger instance
        self.logger = logging.getLogger(__name__)

    def process_files(self) -> None:
        """
        Process all text files in the input directory. For each file, load the customer
        request, process it, and save the results to an output file.
        """
        # Loop over all files in the input directory
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".txt"):
                # Construct the full file path
                full_file_path = os.path.join(self.input_dir, filename)

                try:
                    self.process_file(full_file_path, filename)
                except Exception as e:
                    self.logger.error(
                        f"Failed to process file {filename}. Reason: {e}"
                    )

    def process_file(self, file_path: str, filename: str) -> None:
        """
        Process a single file. Load the customer request, process it, and save the results
        to an output file.

        :param file_path: Path to the file to process.
        :param filename: Name of the file to process.
        """
        # Load the customer request from the text file
        request_text = self.data_processor.load_text_file(file_path)

        # Create a CustomerRequest instance and process the request
        customer_request = CustomerRequest(request_text, self.openai_api)
        customer_request.process_request()

        # Prepare the output data
        output = {
            "customer_request": customer_request.request_text,
            "translated_request": customer_request.translated_text,
            "category": customer_request.category,
            "response": customer_request.response,
            "is_harmful": customer_request.is_harmful,
        }

        # Construct the output file path
        base_filename_without_ext = os.path.splitext(filename)[0]
        output_file_path = os.path.join(
            self.output_dir, f"{base_filename_without_ext}_output.json"
        )

        # Save the output data
        self.data_processor.save_output(output, output_file_path)
