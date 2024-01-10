"""Defines the Book class."""
import re
import os
from typing import OrderedDict
from datetime import datetime
import pdfplumber
import openai
from tqdm import tqdm


class Book():
    """
    Class stores text and provides methods to manipulate it and generate summaries
    """

    def __init__(self, title: str, author: str = "", text: str = ""):
        self.title = title
        self.author = author
        self.text = text
        self.summaries = {}

    def load_pdf(self, pdf_file: str, clean: bool = True):
        """Load text from pdf file"""

        print("Opening PDF file...")
        with pdfplumber.open(pdf_file) as pdf:

            pdf_text = ""

            print("Converting pdf to text...")
            for _page in tqdm(pdf.pages):
                page_text = _page.extract_text()
                if page_text:
                    pdf_text += page_text

            if clean:
                self.text = self.clean_text(pdf_text)
            else:
                self.text = pdf_text

    @staticmethod
    def clean_text(unformatted_text) -> str:
        """
        Clean up text
        """

        print("Cleaning text...")

        if unformatted_text is None:
            return ""

        else:
            # remove unicode
            clean_text = unformatted_text.encode('ascii', 'ignore').decode('ascii')

            # remove repeated whitespace
            clean_text = re.sub(' +', ' ', clean_text)

            # remove numbers from string
            clean_text = re.sub('[0-9]+', '', clean_text)

            # remove multiple returns from string
            clean_text = re.sub('\n+', '\n', clean_text)

            return clean_text

    def get_word_count(self):
        """
        Count words in string
        """
        return len(self.text.split())

    @staticmethod
    def split_text_into_n_word_chunks(text, n_words):
        """
        Split text into chunks of n words
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), n_words):
            chunks.append(' '.join(words[i:i+n_words]))
        return chunks

    @staticmethod
    def tldr_summary(
        input_text:str,
        engine:str="ada",
        summary_prompt:str="\n\ntl;dr:",
        max_tokens:int=64,
        temperature:float=0.1
    ):
        """
        Makes a summarisation text prediction
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            engine=engine,
            prompt=input_text+summary_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.8,
            presence_penalty=0.5,
            # stop=["\n"]
        )
        return response.choices[0].text

    @staticmethod
    def one_word_summary(
        input_text:str,
        engine:str="ada",
    ):
        """
        Makes a summarisation text prediction
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            engine=engine,
            prompt=input_text+"\n\nTo summarise in one word:",
            temperature=0.75,
            max_tokens=3,
            top_p=1,
            frequency_penalty=0.95,
            presence_penalty=1.98,
            # stop=["\n"]
        )
        return response.choices[0].text

    def generate_summary(
        self,
        compression_ratio:float = 0.25,
        min_summary_length:int = 80,
        chunk_length:int = 1000,
        engine:str="ada",
        save:bool=True,
    ):
        """
        Generate a summary for the book
        """
        print("Generating summary...")

        # get word count and et full text as first 'summary'
        word_count = self.get_word_count()
        print(f"Word count: {word_count}...")
        self.summaries[word_count] = self.text

        # check total cost with user
        summary_cost = self.calculate_summary_cost(
            word_count,
            compression_ratio,
            engine=engine
        )
        summary_cost = round(summary_cost, 3)
        cntnue = self.ask_yesno(
            f"\n\nSummary cost using engine {engine}: $US {summary_cost}. Continue? [y/n]"
        )
        if cntnue is False:
            print("\n\nExiting...")
            exit(0)

        # generate shorter and shorter summaries
        while word_count > min_summary_length:

            # get text chunks
            input_text = self.summaries.get(
                min(self.summaries)
            )
            text_chunks = self.split_text_into_n_word_chunks(input_text, chunk_length)

            # append summary of each chunk
            _all_summaries = ""
            max_tokens = max(
                int(min(chunk_length, word_count)*compression_ratio),
                min_summary_length
            )
            for chunk in tqdm(text_chunks):
                _chunk_summary = self.tldr_summary(
                    input_text=chunk,
                    engine=engine,
                    summary_prompt="\n\nIn summary:",
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                _all_summaries += _chunk_summary

            word_count = len(_all_summaries.split())
            self.summaries[word_count] = _all_summaries
            print(f"Generated summary of length {word_count}...")

        accept_one_word = False
        while accept_one_word is False:
            # generate one word summary
            one_word_input_length = min(
                {k: v for k, v in self.summaries.items() if k > 250}
            )  # chose a summary of length 250-1000 words
            one_word_input_text = self.summaries.get(one_word_input_length)
            one_word_summary = self.one_word_summary(
                input_text=one_word_input_text,
                engine=engine
            )
            cost = round(one_word_input_length*0.006/1000, 4)
            accept_one_word = self.ask_yesno(
                f"\n\nAccept '{one_word_summary}' as one-word summary (new estimate costs $US {cost})? [y/n]"
            )

        self.summaries[1] = one_word_summary

        print("Finished generating summaries...!")

        if save:
            self.save_dict_as_markdown(
                self.summaries,
            )

    @staticmethod
    def get_min_dict_key(dictionary):
        """
        Get minimum key in dictionary
        """
        return min(dictionary, key=dictionary.get)

    @staticmethod
    def calculate_summary_cost(total_word_count, compression_ratio, engine):
        """Calculate OPENAI cost"""
        cost_per_1k_token = {
            "ada": 0.0008,
            "curie": 0.0060,
            "babbage": 0.0012,
            "davinci": 0.0600,
        }

        initial_token_count = total_word_count * (1+compression_ratio)
        total_token_count = initial_token_count / (1-compression_ratio)  # infinite geometric series
        total_cost = total_token_count * cost_per_1k_token[engine] / 1000

        return total_cost

    @staticmethod
    def ask_yesno(question):
        """
        Helper to get yes / no answer from user.
        """
        yes = ['yes', 'y']
        no = ['no', 'n'] # pylint: disable=invalid-name

        done = False
        print(question)
        while not done:
            choice = input().lower()
            if choice in yes:
                return True
            elif choice in no:
                return False
            else:
                print("Please respond by yes or no.")

    def save_dict_as_markdown(self, dictionary, filename=None, max_summary_legnth=1000):
        """
        Save dictionary as markdown file
        """
        now = datetime.now().strftime(format="%Y%m%d_%H%M%S")
        filename = filename or f"summaries/{self.title}_Summary_{now}.mD"
        # sort dictionary by key ascending
        sorted_dict = OrderedDict(sorted(dictionary.items()))

        with open(filename, 'w') as file:
            file.write(f"# Summary of {self.title}\n\n")
            for key, value in sorted_dict.items():
                if key <= max_summary_legnth:  # only save summaries of length <= 250 words
                    file.write(f"## {key} word summary\n {value}\n\n")
