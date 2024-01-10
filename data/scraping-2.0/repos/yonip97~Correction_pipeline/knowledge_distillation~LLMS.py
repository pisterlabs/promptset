import time

import openai
import tiktoken
import os
import csv
from general.utils import remove_punctuation

from openai import AzureOpenAI


class LLM_model():
    def __init__(self, temp_save_dir, prompt, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None, **kwargs):
        if prompt is None:
            raise ValueError("prompt can't be None")
        self.past_text_prompt = past_text_prompt
        self.prompt = prompt
        self.model = model
        if model == 'gpt-3.5-turbo':
            openai.api_key = API_KEY
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif model == 'gpt-4':
            openai.api_key = API_KEY
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            openai.api_base = "https://researchopenai2023.openai.azure.com/"
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        elif model == 'gpt-4-turbo':
            self.client= AzureOpenAI(
        api_key=API_KEY,
        api_version='2023-09-01-preview',
        azure_endpoint='https://researchopenai2023eastus2.openai.azure.com/')
        # api_key=os.getenv("OPENAI_API_KEY"),
        # api_version='2023-09-01-preview',
        # azure_endpoint=os.getenv("OPENAI_API_BASE")
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        else:
            raise ValueError(f"model {model} not supported")
        self.open_ai_errors = 0
        self.other_errors = 0
        path_logger = temp_save_dir + '/' + 'logger.txt'
        self.logger = open(path_logger, 'w')

    def call_llm(self, text, max_length, **kwargs):
        input = self.prompt + '\n' + text + '\n' + self.past_text_prompt
        try:
            message = [{
                "role": "user",
                "content": input,
            }]
            if self.model in ['gpt-3.5-turbo','gpt-4']:
                response = openai.ChatCompletion.create(
                    engine=self.model,
                    messages=message,
                    temperature=0,
                    max_tokens=max_length,
                    request_timeout=60
                )
                return response['choices'][0]['message']['content'], None
            elif self.model == 'gpt-4-turbo':
                response = self.client.chat.completions.create(model = self.model,
                                                    messages = message,
                                                    temperature = 0,
                                                    max_tokens = max_length)
                return response.choices[0].message.content, None
            else:
                raise ValueError(f"model {self.model} not supported")
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            self.logger.write(f"Error occurred: {e}")
            self.open_ai_errors += 1
            return None, f"{e}"
        except Exception as e:
            self.other_errors += 1
            self.logger.write(f"Error occurred: {e}")
            print(f"Error in output occurred: {e}")
            return None, f"{e}"


class Summarization_correction_model(LLM_model):
    def __init__(self, temp_save_dir, prompt, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None, **kwargs):
        super(Summarization_correction_model, self).__init__(temp_save_dir, prompt, past_text_prompt, model, API_KEY,
                                                             **kwargs)
        f = open(temp_save_dir + '/' + 'temp_results_summarization.csv', 'w')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow(['text', 'summary', 'revised_summary', 'error'])

    def revise(self, texts, summaries, max_length=None):
        revised_summaries, errors = [], []
        for text, summary in zip(texts, summaries):
            revised_summary, error = self.revise_single(text, summary, max_length=max_length)
            revised_summaries.append(revised_summary)
            errors.append(error)
        return revised_summaries, errors

    def revise_single(self, text, summary, max_length=None):
        text_for_revision = f"Document: \n {text} \n summary: \n {summary} \n"
        if max_length is None:
            max_length = len(self.estimation_tokenizer.encode(summary)) + 10
        revised_summary, error = self.call_llm(text_for_revision, max_length=max_length)
        self.csv_writer.writerow([text, summary, revised_summary, error])
        return revised_summary, error


class LLMFactualityClassifier(LLM_model):
    def __init__(self, temp_save_dir, prompt, text_to_labels, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None,
                 **kwargs):
        super(LLMFactualityClassifier, self).__init__(temp_save_dir, prompt, past_text_prompt, model, API_KEY, **kwargs)
        f = open(temp_save_dir + '/' + 'temp_results_classification.csv', 'w')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow(['text', 'summary', 'prediction_text', 'prediction_label', 'error'])
        self.text_to_labels = text_to_labels

    def classify(self, texts, summaries, max_length):
        predictions, errors = [], []
        for i in range(len(texts)):
            document = texts[i]
            summary = summaries[i]
            prediction, error = self.classify_single(document, summary, max_length)
            predictions.append(prediction)
            errors.append(error)
        return predictions, errors

    def classify_single(self, text, summary, max_length):
        text = f"Document: \n {text} \n summary: \n {summary} \n"
        response, error = self.call_llm(text, max_length=max_length)
        if response is not None:
            response = remove_punctuation(response.lower().strip())
        if response not in self.text_to_labels:
            prediction = None
        else:
            prediction = self.text_to_labels[response]
        self.csv_writer.writerow([text, summary, response, prediction, error])
        return prediction, error
