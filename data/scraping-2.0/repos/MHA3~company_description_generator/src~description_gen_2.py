import re
import logging
import os
import pandas as pd
import time
import joblib
from tqdm import tqdm
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants for environment and file paths
ENV = "dev"  # Or "prod" for production

# File paths are chosen based on the environment
INPUT_FILE_PATH = './data/task_sources.csv' if ENV == "prod" else './data/input_sample.csv'
OUTPUT_FILE_PATH = './data/company_descriptions.csv' if ENV == "prod" else './data/out_langchain.csv'

# Ensure the necessary directories exist
os.makedirs(os.path.dirname(INPUT_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

class CompanyDescriptionProcessor:
    LLM_NAME = "gpt-4"
    OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
    LLM = ChatOpenAI(temperature=0.3, model=LLM_NAME, openai_api_key=OPEN_AI_KEY)
    INCOMPLETE_INFO_ERROR = "ERROR: could not infer the required information from the given text."
    _models = {}

    def __init__(self, batch_size=10, cache_dir='translation_cache_dir'):
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    @classmethod
    def _load_translation_models(cls, source_lang):
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'
        if model_name not in cls._models:
            try:
                cls._models[model_name] = {
                    'model': MarianMTModel.from_pretrained(model_name),
                    'tokenizer': MarianTokenizer.from_pretrained(model_name)
                }
            except Exception as e:
                logging.error(f"Failed to load model {model_name}: {e}")
                raise

    def clean_description(self, text):
        cleaned = re.sub('<[^<]+?>', '', text)
        cleaned = re.sub(r'http\S+', '', cleaned)
        cleaned = re.sub(' +', ' ', cleaned)
        return cleaned.strip()

    def get_cache(self, source_lang):
        cache_file = os.path.join(self.cache_dir, f'{source_lang}_cache.pkl')
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
        return {}

    def set_cache(self, cache, source_lang):
        cache_file = os.path.join(self.cache_dir, f'{source_lang}_cache.pkl')
        joblib.dump(cache, cache_file)

    def translate_to_english(self, text, source_lang):
        # If the source language is already English, skip translation
        if source_lang == 'en':
            return text

        cache = self.get_cache(source_lang)
        if text in cache:
            return cache[text]

        try:
            self._load_translation_models(source_lang)
            model = self._models[f'Helsinki-NLP/opus-mt-{source_lang}-en'][
                'model']
            tokenizer = self._models[f'Helsinki-NLP/opus-mt-{source_lang}-en'][
                'tokenizer']

            tokenized_text = tokenizer.encode(text, return_tensors="pt",
                                              truncation=True)
            translation = model.generate(tokenized_text)
            translated_text = tokenizer.decode(translation[0],
                                               skip_special_tokens=True)

            cache[text] = translated_text
            self.set_cache(cache, source_lang)
        except Exception as e:
            logging.warning(
                f"No translation model available for {source_lang}-en. Keeping original text.")
            translated_text = text

        return translated_text

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def preprocess_csv(self, file_path):
        df = pd.read_csv(file_path)
        df['cleaned_description'] = df['text'].apply(self.clean_description)

        if 'language' not in df.columns:
            tqdm.pandas(desc="Detecting Language")
            df['language'] = df['cleaned_description'].progress_apply(self.detect_language)

        tqdm.pandas(desc="Translating")
        df['preprocessed_description'] = df.progress_apply(lambda row: self.translate_to_english(row['cleaned_description'], row['language']), axis=1)

        df.to_csv("./data/translated_descriptions.csv", index=False)
        return df[['preprocessed_description']]

    def process_text(self, index, text, chain):
        """Helper function to process text using specified chain."""
        if len(text) < 50:
            return index, self.INCOMPLETE_INFO_ERROR
        return index, chain.run(text)

    def execute_chain_in_parallel(self, df_column, chain, max_rate=10):
        """Execute a specified chain function in parallel on a DataFrame column."""
        results_list = [None] * len(df_column)  # Predefine the list to store the results
        with ThreadPoolExecutor(max_workers=max_rate) as executor:
            # Start time to ensure rate limit
            start_time = time.time()

            # Prepare a list to hold future tasks
            futures_to_index = {
                executor.submit(self.process_text, index, text, chain): index
                for index, text in enumerate(df_column)
            }

            # Process futures as they complete
            for future in as_completed(futures_to_index):
                index = futures_to_index[future]  # Retrieve the original index for the result
                try:
                    # Attempt to get the result of the API call
                    _, result = future.result()
                    results_list[index] = result
                except Exception as e:
                    logging.error(f"An error occurred at index {index}: {e}")
                    # In case of an error, you could store a default value or error indicator
                    results_list[index] = 'Error: Unable to process text.'

            # Calculate the time taken to process the requests
            elapsed_time = time.time() - start_time

            # If the requests were processed in less than a minute, sleep for the remaining time
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)

        return results_list

    def generate_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates summaries from the cleaned and translated text in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the cleaned and translated text.

        Returns:
        pd.DataFrame: A DataFrame containing the generated summaries.
        """
        # Set up summarizing chain
        summarizing_prompt = ChatPromptTemplate.from_template(
            "Extract summary in the format enclosed by single quotes"
            "'PROBLEM: describe the problem the company is trying to solve "
            "SOLUTION: company's proposed solution "
            "TARGET USERS: target users of the company "
            "OTHER DETAILS: other important details of the company', "
            "for the following company description enclosed by triple backticks "
            "```{company_description}```."
            "If the information is not available then return a message like this:"
            f"'{self.INCOMPLETE_INFO_ERROR}'"
        )
        summarizing_chain = LLMChain(llm=self.LLM, prompt=summarizing_prompt)

        # Generate summaries
        summaries = self.execute_chain_in_parallel(df["preprocessed_description"], summarizing_chain)

        # Store results
        df["summaries"] = summaries
        return df[["summaries"]]


if __name__ == "__main__":
    print("Initializing")
    processor = CompanyDescriptionProcessor()

    print("preprocessing")
    clean_df = processor.preprocess_csv(INPUT_FILE_PATH)
    print(clean_df)
    print("finished preprocessing")

    print("summarizing")
    summaries_df = processor.generate_summaries(clean_df)
    print("finished summarizing")

    summaries_df.to_csv(OUTPUT_FILE_PATH, index=False)



