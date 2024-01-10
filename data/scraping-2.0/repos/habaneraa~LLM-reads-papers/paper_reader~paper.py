
import re
import json
from typing import List, Callable

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .llm import chatgpt
from .prompt_templates import section_titles_identification, section_summarization
from .tokens_counter import tiktoken_length, count_message_tokens


# currently our splitting strategy is dividing the text by section titles
# here are regex patterns for recognizing top level section titles
# they are NOT general enough to match all kinds of title texts!
# please provide specific regex if the following ones are not working
pattern_digits_dot = re.compile(r'^\d+\.\s+.+')
pattern_digits = re.compile(r'^\d+\s+.+')
pattern_romes = re.compile(r'^[IVXLCDM]+\.?\s+.+')
used_pattern = pattern_digits


class Paper:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=tiktoken_length,
    )

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.pdf_obj = PdfReader(filepath)
        # self.plain_text = ''
        # for page in self.pdf_obj.pages:
        #     self.plain_text += page.extract_text()
        self.plain_text = extract_text(filepath)

        self.paper_metadata = self.pdf_obj.metadata
        self.paper_summaries = {}
        self.paper_parts = None

    def parse_pdf_title(self) -> List[str]:
        possible_titles = []
        def check_text_block(elem):
            txts = elem.get_text().split('\n')
            for t in txts:
                if used_pattern.match(t):
                    possible_titles.append(t)

        for page_layout in extract_pages(self.filepath):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    check_text_block(element)

        response = chatgpt(section_titles_identification.format_messages(
            list_of_texts=possible_titles
            ))
        return json.loads(response.content)['titles']

    def split_paper_by_titles(self, titles: List[str]) -> None:
        title_positions = []
        for title in titles:
            position = self.plain_text.find(title)
            if position != -1:
                title_positions.append((position, title))

        title_positions.sort()

        paper_parts = []
        for i, (position, title) in enumerate(title_positions):
            start_pos = position
            end_pos = title_positions[i+1][0] if i < len(title_positions) - 1 else len(self.plain_text)
            paper_part = self.plain_text[start_pos:end_pos].strip()
            paper_parts.append((title, paper_part))

        self.paper_parts = paper_parts
    
    def estimate_cost(self, key_points: List[str], dollar_per_k_tokens=0.002):
        prompt_tokens = 0
        formatted_key_points = '\n'.join([f'{i+1}. {p.strip()}' for i, p in enumerate(key_points)])
        for (title, contents) in self.paper_parts:
            msgs = section_summarization.format_messages(
                key_points=formatted_key_points,
                section_title=title,
                section_texts=contents,
            )
            msgs, _ = chatgpt._create_message_dicts(msgs, None)
            tokens = count_message_tokens(msgs)
            # if tokens > 2000:
            #     print(f'Warning: section \"{title}\" exceeds 2000 input tokens ({tokens}).')
            prompt_tokens += tokens
        estimate_tokens = len(self.paper_parts) * 100 + prompt_tokens
        print('Total prompt tokens:', prompt_tokens)
        print('Estimate cost:', estimate_tokens*dollar_per_k_tokens/1000)

    def read_paper(self, key_points: List[str], callback: Callable = None) -> None:
        if self.paper_parts is None:
            titles = self.parse_pdf_title()
            self.split_paper_by_titles(titles)

        formatted_key_points = '\n'.join([f'{i+1}. {p.strip()}' for i, p in enumerate(key_points)])
        # Reading and summarizing each part of the research paper
        for (title, contents) in self.paper_parts:
            chunks = self.text_splitter.split_text(contents)
            section_summary = ''
            for i, chunk in enumerate(chunks):
                response = chatgpt(section_summarization.format_messages(
                    key_points=formatted_key_points,
                    section_title=title,
                    section_texts=chunk,
                ))
                if callback:
                    callback(response.content, title, i, len(chunks))
                section_summary += response.content + '\n\n'
            self.paper_summaries[title] = section_summary
