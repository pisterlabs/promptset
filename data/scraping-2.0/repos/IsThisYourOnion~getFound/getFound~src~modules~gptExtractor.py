import os
import openai
import json
from getFound.src.config import params
from getFound.src.utils.utils import DataManager, clean_text


class TextProcessor:
    def __init__(self, api_key, full_text_corpus, model="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.full_text_corpus = full_text_corpus
        self.model = model
        self.prompt = params.gpt_prompt

    def _chunk_text(self, text, length):
        """Divide a text into chunks of a specific length"""
        return [text[i: i + length] for i in range(0, len(text), length)]

    def _extract_data(self, text_chunk):
        """Run the GPT-3 model to process a text chunk and return the completion"""
        completion = openai.ChatCompletion.create(
          model=self.model,
          messages=[
            {"role": "user", "content": f"{self.prompt}{text_chunk}"}
          ]
        )
        return completion.choices[0].message.content

    def process_text(self):
        """Divide the text into chunks and process each chunk"""
        text_chunks = self._chunk_text(self.full_text_corpus, 750)
        results = []
        for chunk in text_chunks:
            result = self._extract_data(chunk)
            results.append(result)

        return results


def GPTKeywords():
    manager = DataManager()
    text = manager.read_data('raw_job_text', 'txt', False)
    phrases = []
    combined_string = ' '.join(text)
    cleaned_combined_string = clean_text(combined_string)
    # usage example
    processor = TextProcessor(params.open_ai_key, cleaned_combined_string)
    processed_results = processor.process_text()  # process the text and get results
    manager.write_data('job_keywords', 'extracted_keywords', 'json', processed_results)  # write results into a json file
    return processed_results  # if you want to return the processed results

