import json

from image_analyzer import OpenAIAnalyzer

"""
This module provides a class for processing multiple images by analyzing them using the OpenAI API. It utilizes an image encoder for base64 encoding and an image analyzer for communicating with the OpenAI API.

Classes:
    ImageProcessor: Orchestrates the process of encoding and analyzing images.

Dependencies:
    - OpenAIAnalyzer from image_analyzer module: Used to interact with the OpenAI API for image analysis.

ImageProcessor:
    A class that integrates the image encoding and analyzing functionalities to process multiple images.

    Methods:
        __init__(api_key): Initializes the ImageProcessor with an API key for the OpenAIAnalyzer.
        process_image(base64_image, location): Processes a list of image paths and analyzes them based on the specified location type ('interior' or 'exterior').

            Args:
                base64_image (str): A base64 encoded string of the image to be analyzed.
                location (str): A string indicating the location type for the analysis ('interior' or 'exterior').

            Returns:
                list: A list of dictionaries with the results of the analysis. Each dictionary includes an 'http_status' key indicating the result status (200 for valid, 403 for rejection) and, if applicable, a 'reason' key with details of the rejection.

            Raises:
                json.JSONDecodeError: If the response from the OpenAI API is not a valid JSON.
                Exception: For any other unexpected errors during the process.

Usage:
    An instance of ImageProcessor is created by providing an API key. The `process_image` method is then used to process a list of image paths. The method encodes each image in base64, sends it for analysis, and compiles the results, handling different response scenarios and potential errors.
"""


class ImageProcessor:
    def __init__(self, api_key):
        self.analyzer = OpenAIAnalyzer(api_key)

    def process_image(self, base64_image, location):
        results = []
        try:
            analysis_result = self.analyzer.analyze_image(base64_image, location)
            text = analysis_result['choices'][0]['message']['content']

            if text.strip() == "valid":
                results.append({"http_status": 200})
            else:
                reason_list = json.loads(text)
                results.append({"http_status": 403, "reason": reason_list})
        except json.JSONDecodeError as e:
            results.append({'error': 'Invalid JSON response'})
        except Exception as e:
            results.append({'error': 'Unexpected error occurred'})
        return results[0]
