# src.text_processor.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from src.base_models import Article, Category
import re
import os
import requests
import json
import tiktoken
import numpy as np
from langchain.text_splitter import (TokenTextSplitter,
                                     CharacterTextSplitter,
                                     SpacyTextSplitter)

from config import ner_pattern, entity_ruler_config, non_fuzzy_list
from spacy.lang.en import English


def convert_fuzzy_match(patterns: List[Dict], non_fuzzy_list: List[str]) -> List[Dict]:
    spacy_patterns = []
    for entry in patterns:
        label = entry['label']
        pattern = entry['pattern']

        if isinstance(entry['pattern'], str):
            if pattern in non_fuzzy_list: token_patterns = [{'LOWER': word.lower()} for word in entry['pattern'].split()]
            else: token_patterns = [{'LOWER': {'FUZZY1': word.lower()}} for word in entry['pattern'].split()]
        else:
            if pattern in non_fuzzy_list: token_patterns = [{'LOWER': word.lower()} for word in entry['pattern'].split()]
            else: token_patterns = [{'LOWER': {'FUZZY1': token['LOWER']}} for token in entry['pattern']]
        spacy_patterns.append({'label': label, 'pattern': token_patterns})
    return spacy_patterns


class TextProcessor(ABC):

    @abstractmethod
    def build_section_dict(self, article, exclude_sections):
        pass

    @abstractmethod
    def remove_curly_brackets(self, section_dict):
        pass

    @abstractmethod
    def build_embeddings(self, article) -> Dict:
        pass

    @abstractmethod
    def build_training_metadata(self, article, pattern):
        pass

    @abstractmethod
    def build_token_len_dict(self, article):
        pass

    @abstractmethod
    def extract_headings(self, article):
        pass

    @abstractmethod
    def chunk_text_dict(self, section_dict):
        pass

    @abstractmethod
    def build_metadata_from_model(self, article, model):
         pass


class BaseTextProcessor(TextProcessor):
    """
    Need some way to split the returned text of tokenize and clean.
    t&c -> chunk -> hf_clean -> vectors
    [1] -> [many] -> [many cleaned] -> [[f32],[f32],[f32]]
    """

    def extract_headings(self, article):
        normalized_text = re.sub(r'={3,}', '==', article.text)
        section_pattern = r'(==\s*[^=]+?\s*==)'
        headings = re.findall(section_pattern, normalized_text, re.MULTILINE)
        print("Extracting headings...")
        for heading in headings:
            print(heading)
        if not headings:
            print("No major headings found.")

    def build_section_dict(self, article: Article, exclude_sections: List[str]) -> Dict[str, str]:
        """
        Split on section headings and exclude specified headings.
        :param article: Article object containing the text.
        :param exclude_sections: List of headings to exclude.
        :return: Dictionary with section headings as keys and text as value.
        """
        normalized_text = re.sub(r'={3,}', '==', article.text)
        # normalized_text = re.sub(r'\{\{[^}]*?\}\}', '', normalized_text)
        section_pattern = r'(==\s*[^=]+?\s*==)'
        parts = re.split(section_pattern, normalized_text)
        sections = {'Introduction': parts[0].strip()}

        for i in range(1, len(parts), 2):
            section_title = parts[i].strip("= ").strip()

            if section_title not in exclude_sections:
                sections[section_title] = parts[i + 1].strip()

        return sections

    def remove_curly_brackets(self, section_dict) -> Dict:
        cleaned_sections = {}

        for section, text in section_dict.items():
            stack = []
            to_remove = []
            text_list = list(text)

            for i, char in enumerate(text_list):
                if char == '{':
                    stack.append(i)
                elif char == '}':
                    if stack:
                        start = stack.pop()
                        if not stack:
                            to_remove.append((start, i))

            for start, end in reversed(to_remove):
                del text_list[start:end + 1]

            cleaned_sections[section] = ''.join(text_list)

        return cleaned_sections

    def chunk_text_dict(self, cleaned_section_dict) -> Dict:
        # tiktoken tokenizer
        # Could use Langchain textSplitter
        sectioned_dict = {}
        avg_chunk_len = []

        text_splitter = SpacyTextSplitter(chunk_size=1200, chunk_overlap=20)
        # Arbitrary chunk size and overlap. also arbitrary splitter
        encoding = tiktoken.get_encoding("cl100k_base")
        for key, value in cleaned_section_dict.items():
            chunked_text = text_splitter.split_text(value)
            for idx, chunk in enumerate(chunked_text):
                tokens_integer = encoding.encode(chunk)
                avg_chunk_len.append(len(tokens_integer))
                # print(f'key: {key}_{idx}, len: {len(tokens_integer)}')

                # print(f'Token Length: {len(tokens_integer)} at idx {idx} of {key}')
                sectioned_dict[f'{key}_{idx}'] = chunk
        # print(f'Average Chunk Length: {np.mean(avg_chunk_len)}')
        return sectioned_dict

    # def recursive_chunk_text_dict(self, article: Article, cleaned_section_dict) -> Dict:
    #     # tiktoken tokenizer
    #     # Could use Langchain textSplitter
    #     sectioned_dict = {}
    #     text_splitter = TokenTextSplitter(chunk_size=460, chunk_overlap=10)
    #     # Arbitrary chunk size and overlap. also arbitrary splitter
    #     encoding = tiktoken.get_encoding("cl100k_base")
    #     for key, value in cleaned_section_dict.items():
    #
    #         # split function here
    #         for idx, chunk in enumerate(chunked_text):
    #             tokens_integer = encoding.encode(chunk)
    #             #print(f'Token Length: {len(tokens_integer)} at idx {idx} of {key}')
    #             print(f'{key}_{idx}')
    #             sectioned_dict[f'{key}_{idx}'] = chunk
    #     return sectioned_dict

    def build_token_len_dict(self, article: Article) -> str:
        """
        Idea is to tokenize and split content of articles. And to estimate cost of the article
        :param article: Article object with built text_dict
        :return: Article Object with text_dict edited in place
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        len_dict = {}
        for section, content in article.text_dict.items():
            num_tokens = len(encoding.encode(content))
            if num_tokens > 10:
                len_dict[section] = num_tokens
            else:
                print(section, num_tokens, 'insufficient_token_section')
        return f'Tokens in article: {sum(i for i in len_dict.values())}'

    def build_embeddings(self, article: Article) -> Dict:
        # URL of your Flask API endpoint
        api_url = "http://127.0.0.1:5000/embeddings_api"
        payload = {'article': article.text_dict}
        # Make the POST request
        response = requests.post(api_url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            embed_dict = response.json()
            return embed_dict
        else:
            print("Error in Embedding Encoding API call:", response.status_code, response.text)

    def build_training_metadata(self, article: Article, pattern):
        """
        Method: Matches patterns to text and builds the return data structure.
        :param non_fuzzy_list:
        :param article: Article Object w/ attribute text_dict. {heading_0: text:str, heading_1: text:str}
        :param  pattern: List[Dict] = [
            {"label": "Probability & Statistics", "pattern": "normal distribution"},
            {"label": "Probability & Statistics", "pattern": "continuous distribution"}]
        :return List as follows:
        data = [
        ("This is text with important information",[(start_span, end_span, label)]),
        ("important information and I promise it is important",[(start_span, end_span, label), (start_span, end_span, label)])
        ]
        """
        concatenated_text = ' '.join([section_text for section_text in article.text_dict.values()])
        cleaned_text = re.sub(r'[\n\t]+', ' ', concatenated_text)

        nlp = English()

        # Instead of creating a component and then adding it, directly add the component by its name
        ruler = nlp.add_pipe("entity_ruler", config=entity_ruler_config)
        ruler.add_patterns(pattern)

        # Remove the redundant `nlp.add_pipe(ruler)` line
        doc = nlp(cleaned_text)
        entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents]
        return cleaned_text, entities

    def build_metadata_from_model(self, article: Article, model):
        """
        Method: Matches patterns to text and builds the return data structure.
        :param article: Article Object w/ attribute text_dict. {heading_0: text:str, heading_1: text:str}
        :param  model: The loaded fine-tuned model
        :return List as follows:
        data = [
        ("This is text with important information",[(start_span, end_span, label)]),
        ("important information and I promise it is important",[(start_span, end_span, label), (start_span, end_span, label)])
        ]
        """
        # test_pattern_fuzzy = convert_fuzzy_match(pattern)
        # metadata_dict = {}
        # nlp = English()
        # ruler = nlp.add_pipe("entity_ruler", config=entity_ruler_config)
        # ruler.add_patterns(test_pattern_fuzzy)
        #
        # for heading, content in article.text_dict.items():
        #     sub_metadata_dict = {}
        #     doc = nlp(content)
        #     for ent in doc.ents:
        #         if ent.label_ in sub_metadata_dict:
        #             sub_metadata_dict[ent.label_].add(ent.text)
        #         else:
        #             sub_metadata_dict[ent.label_] = {ent.text}
        #     metadata_dict[heading] = {key: list(value) for key, value in sub_metadata_dict.items()}
        # return metadata_dict
        """This section will be used to call the ner endpoint when model is developed. The model will be
        called and returned in a similar data structure to maintain record integrity to the above loop"""
        pass
