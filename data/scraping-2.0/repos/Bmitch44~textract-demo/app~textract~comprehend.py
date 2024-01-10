"""
In this file I will begin trying to comprehend the output of the textract, I will experiemnt with multiple 
tools including GPT-3 and GPT-4, and any other tools that I can find.
"""

import json
from app.textract.ctrp import Document
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class TextExtractor:
    def __init__(self, json_path):
        self.json_path = json_path

    def extract_text(self):
        """Extracts the text from the json file using Document class"""
        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"An error occurred while opening the file: {e}")
            return None

        doc = Document(data)
        text = ""

        for page in doc.pages:
            for content in page.content:
                text += self._process_content(content)
        return text

    def _process_content(self, content):
        """Processes the content based on its type"""
        text = ""
        if content.blockType == "SELECTION_ELEMENT":
            text += self._process_selection_element(content)
        elif content.blockType == "KEY_VALUE_SET":
            text += self._process_key_value_set(content)
        elif content.blockType == "TABLE":
            text += self._process_table(content)
        elif content.blockType == "LINE":
            text += self._process_line(content)
        return text

    def _process_selection_element(self, content):
        """Processes SELECTION_ELEMENT"""
        return f"\nSELECTION_ELEMENT:\n" + content.selectionStatus + " "

    def _process_key_value_set(self, content):
        """Processes KEY_VALUE_SET"""
        text = f"\nKEY_VALUE_SET:\n"
        text += "KEY: " + (content.key.text + " " if content.key is not None else "")
        text += "\nVALUE: " + (content.value.text + " " if content.value is not None else "")
        return text

    def _process_table(self, content):
        """Processes TABLE"""
        text = f"\nTABLE:\n"
        for i, row in enumerate(content.rows):
            for j, cell in enumerate(row.cells):
                text += f"Row {str(i)} - Column {str(j)}: " + cell.text + " "
            text += "\n"
        return text

    def _process_line(self, content):
        """Processes LINE"""
        return f"\nLINE:\n" + content.text + " "


class TextComprehender:
    def __init__(self):
        self.messages = []

    def comprehend_text(self, text, init=False):
        """Comprehends the text using GPT-3"""
        initial_messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about the given text which\
                  represents a PDF document on geological data found in Newfoundland and Labrador. You must give a structured\
                  representation of the given text."},
                {"role": "user", "content": "Give an overview of the following text which represents a PDF document:\n" + text}
            ]
        
        if init:
            self.messages = initial_messages
        else:
            self.messages.append({"role": "user", "content": text})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=self.messages,
        )
        
        return self._process_completion(completion)
    
    def _process_completion(self, completion):
        """Processes the completion"""
        message = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": message})
        return message


if __name__ == '__main__':
    extractor = TextExtractor('app/results/textract_results/corrected_test_cropped_output.json')
    text = extractor.extract_text()
    comprehender = TextComprehender()
    print(comprehender.comprehend_text(text))

