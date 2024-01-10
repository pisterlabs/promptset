from dataclasses import dataclass

from more_itertools import batched
from typing import Sequence, Collection, Tuple
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import openai
import os
import json
import torch as th
import numpy as np
import random
import time


@dataclass
class Page:
    text: str
    idx: int


@dataclass
class Question:
    question: str
    options: Sequence[str]
    answer: int
    page_indices: Sequence[int]

@dataclass
class Flashcard:
    front: str
    back: str
    page_indices: Sequence[int]


@dataclass
class Activity:
    name: str


@dataclass
class Summary(Activity):
    content: str


@dataclass
class Quiz(Activity):
    questions: Sequence[Question]


@dataclass
class Flashcards(Activity):
    cards: Sequence[Flashcard]


@dataclass
class Chapter:
    title: str
    pages: Sequence[Page]
    activities: Collection[Activity]
    embeddings: np.ndarray


@dataclass
class ToCEntry:
    title: str
    start_page: int
    end_page: int  # non-inclusive


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


class Textbook:
    MODEL = 'gpt-3.5-turbo-0613'

    def __init__(self, textbook: Sequence[str], embeddings: np.ndarray | None = None) -> None:
        self.raw_textbook: Sequence[str] = textbook
        table_of_contents_raw: Sequence[str] = self._extract_table_of_contents()  # pages representing the table of contents
        sections: Sequence[str] = self._get_sections(table_of_contents_raw)  # the sections of the textbook
        segment: Tuple[Sequence[ToCEntry], Sequence[np.ndarray]] = self._segment(sections, embeddings)  # get chapters and their associated pages
        self.table_of_contents: Sequence[ToCEntry] = segment[0]  # get the table of contents
        self.embeddings: Sequence[np.ndarray] = segment[1]  # get the embeddings for each page

        print('Generating chapters...')
        self.chapters: Sequence[Chapter] = self._parse_chapters(self.table_of_contents, self.embeddings)  # get the actual chapters
    
    @staticmethod
    def _count_numbers(page: str) -> int:  # count the number of separate instances of numbers in the page; i.e. 34 34 1 has 3 numbers
        tokens = page.split()
        numbers = [token for token in tokens if token.isnumeric()]
        return len(numbers)
    
    def _extract_table_of_contents(self) -> Sequence[str]:
        start_keywords = ['table of contents', 'contents']
        end_condition = lambda page: Textbook._count_numbers(page) < 10  # if there are less than 10 numbers on the page, we assume it's the end of the table of contents

        start_page_number = -1
        end_page_number = -1
        is_in_table_of_contents = False
        for i, page in enumerate(self.raw_textbook):
            if end_page_number != -1:
                break

            if is_in_table_of_contents:
                if end_condition(page.lower()):
                    end_page_number = i
                    break
            else:
                for start_keyword in start_keywords:
                    if start_keyword in page.lower():
                        is_in_table_of_contents = True
                        start_page_number = i
                        break
        else:
            end_page_number = len(self.raw_textbook)

        return self.raw_textbook[start_page_number:end_page_number]
    
    def _get_sections(self, table_of_contents: Sequence[str]) -> int:
        messages = [
            {
                'role': 'user',
                'content': 'Can you output the names of the sections of this textbook in order based on the table of contents? I will send a message for each page in the table of contents. It may be a little wrong and numbers before the section names may be inaccurate, but the order is correct. There may be some hierarchies in the table of contents; PLEASE ignore those, simply output the bottom-level sections. ONLY include one level of sections; if there are chapters organized into subchapters, IGNORE then chapters and only output the subchapters. There would be more subchapters than chapters. Also, EXCLUDE the preface/introduction/table of contents and the bibliography/index.'
            },
            *[{
                'role': 'user',
                'content': page
            } for page in table_of_contents]
        ]

        SECTION_FUNCTION_NAME = 'set_sections'

        count_function = {
            'name': SECTION_FUNCTION_NAME,
            'description': 'Sets the section titles in a textbook based on the table of contents.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sections': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                        },
                        'description': 'The titles of the sections in the textbook, in order.'
                    },
                },
                'required': ['sections'],
            },
        }

        response = openai.ChatCompletion.create(
            model=self.MODEL,
            messages=messages,
            functions=[count_function],
            function_call={'name': SECTION_FUNCTION_NAME},
        )

        arguments = json.loads(response['choices'][0]['message']['function_call']['arguments'])

        predicted_sections = arguments['sections']
        predicted_sections.insert(0, 'Preface/Introduction/Table of Contents')  # add the preface/introduction/table of contents as the first section   
        predicted_sections.append('Bibliography/Index')  # add the bibliography/index as the last section

        return predicted_sections
    
    def _segment(self, sections: Sequence[str], embeddings: np.ndarray | None = None) -> Tuple[Sequence[ToCEntry], Sequence[np.ndarray]]:
        INDEX_DISTANCE_WEIGHT = 5  # how much to weight the index of the page as a distance metric
        EMBEDDING_BATCH_SIZE = 64  # the batch size to use when generating embeddings
        BATCH_WAIT = 0.5  # the amount of time to wait between batches (😡)

        if embeddings is None:
            raw_embeddings = []
            for batch in batched(self.raw_textbook, EMBEDDING_BATCH_SIZE):
                current_embeddings = openai.Embedding.create(
                    model='text-embedding-ada-002',
                    input=batch,
                )['data']
                raw_embeddings.extend([embedding['embedding'] for embedding in current_embeddings])

                time.sleep(BATCH_WAIT)  # stupid ass bitch ass rate limit
            
            embeddings = np.array(raw_embeddings)
        
        # PCA decomposition of embeddings
        PCA_DIM = 32
        
        pca = PCA(n_components=PCA_DIM)
        pca_result = pca.fit_transform(embeddings)
        embeddings = pca_result / np.expand_dims(np.linalg.norm(pca_result, axis=1), axis=1)

        raw_embeddings = embeddings
        embeddings = np.empty((embeddings.shape[0], embeddings.shape[1] + 1))
        embeddings[:, :-1] = raw_embeddings
        embeddings[:, -1] = np.linspace(0, INDEX_DISTANCE_WEIGHT, raw_embeddings.shape[0])

        section_title_embeddings = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=sections,
        )['data']
        section_title_embeddings = np.array([[*embedding['embedding'], 0] for i, embedding in enumerate(section_title_embeddings)])

        # Split up the embeddings into num_sections initial clusters
        cluster_indices = th.linspace(0, len(embeddings), len(sections) + 1).long()
        cluster_index_centers = th.linspace(0, INDEX_DISTANCE_WEIGHT, len(sections)).long()
        cluster_centers = []

        for i, (start, stop) in enumerate(zip(cluster_indices[:-1], cluster_indices[1:])):
            current_embedding = embeddings[start:stop].mean(axis=0)
            current_embedding[-1] = 0
            current_embedding /= np.linalg.norm(current_embedding)
            current_embedding[-1] = cluster_index_centers[i]

            cluster_centers.append(current_embedding)
        
        cluster_centers = np.stack(cluster_centers, axis=0)

        kmeans = KMeans(n_clusters=len(sections), init=cluster_centers, random_state=0).fit(embeddings)

        chapter_assignments = kmeans.labels_
        toc_entries = [ToCEntry(title=section_title, start_page=0, end_page=0) for section_title in sections]
        split_embeddings = [None] * len(sections)

        for i in range(len(sections)):
            page_indices = np.where(chapter_assignments == i, np.arange(len(chapter_assignments)), -1)
            page_indices = page_indices[page_indices != -1]
            toc_entries[i].start_page = page_indices.min()
            toc_entries[i].end_page = page_indices.max() + 1
            split_embeddings[i] = embeddings[page_indices]
        
        return toc_entries, split_embeddings
        
        
    def _parse_chapters(self, table_of_contents: Sequence[ToCEntry], embeddings: Sequence[np.ndarray]) -> Sequence[Chapter]:
        chapters = []

        print(len(table_of_contents))
        for toc_entry, embedding in tqdm(zip(table_of_contents, embeddings), total=len(table_of_contents)):
            start_page = toc_entry.start_page
            end_page = toc_entry.end_page
            chapter_title = toc_entry.title

            # Extract the content of the chapter between start_page and end_page
            chapter_content = [Page(text=page_content, idx=i + start_page) for i, page_content in enumerate(self.raw_textbook[start_page:end_page])]

            # Create a Chapter object and add it to the chapters list
            chapter = Chapter(title=chapter_title, pages=chapter_content, activities=self._generate_activities(chapter_title, chapter_content), embeddings=embedding)  # TODO: fill activities later
            chapters.append(chapter)

        return chapters
    
    def _generate_activities(self, chapter_title: str, chapter_content: Sequence[Page]) -> Sequence[Activity]:
        activities = []

        # Generate a summary for the chapter
        summary = self._generate_summary(chapter_title, chapter_content)
        activities.append(summary)

        # Generate a quiz for the chapter
        quiz = self._generate_quiz(chapter_title, chapter_content)
        activities.append(quiz)

        # Generate flashcards for the chapter
        flashcards = self._generate_flashcards(chapter_title, chapter_content)
        activities.append(flashcards)

        return activities
        
    def _generate_summary(self, title: str, pages: Sequence[Page]) -> Summary: 
        summary = self._generate_summary_recursive([page.text for page in pages])

        return Summary(name='Section Summary', content=summary)

    def _generate_summary_recursive(self, pages: Sequence[str]) -> str:
        if len(pages) == 1:
            return pages[0]
        
        BATCH_LEN_THRESHOLD = 4_000  # the threshold for the length of a batch of pages
        
        batches = []
        current_batch = []
        current_batch_len = 0

        for page in pages:
            current_batch.append(page)
            current_batch_len += len(page)

            if current_batch_len > BATCH_LEN_THRESHOLD:
                batches.append(current_batch)
                current_batch = []
                current_batch_len = 0
        
        if len(current_batch) > 0:
            batches.append(current_batch)
        
        summaries = [self._generate_summary_of_strings(batch) for batch in batches]
        
        return self._generate_summary_recursive(summaries)
    
    def _generate_summary_of_strings(self, strings: Sequence[str]) -> str:
        messages = [
            {
                'role': 'user',
                'content': 'Please generate a summary of the following:'
            },
            *[{
                'role': 'user',
                'content': string
            } for string in strings]
        ]

        response = openai.ChatCompletion.create(
            model=self.MODEL,
            messages=messages,
        )

        summary = response['choices'][0]['message']['content']

        return summary

    def _generate_quiz(self, title: str, pages: Sequence[Page]) -> Quiz: 
        questions = []
        batch_size = 3  # Set the batch size to 3 pages

        for batch in tqdm(batched(pages, batch_size)):
            # Generate a question for the chunk of pages
            question = self._generate_question(title, batch)
            questions.append(question)

        return Quiz(name='Section Quiz', questions=questions)

    def _generate_question(self, title: str, pages: Sequence[Page]) -> Question:
        messages = [
            {
                'role': 'user',
                'content': 'Please create a multiple-choice question from the information in these textbook pages with 3 incorrect answers and 1 correct answer.'
            },
            *[{
                'role': 'user',
                'content': page.text
            } for page in pages]
        ]

        QUIZ_FUNCTION_NAME = 'create_question'

        quiz_function = {
            'name': QUIZ_FUNCTION_NAME,
            'description': 'Displays a multiple-choice question',
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': 'The question to ask.'
                    },
                    'incorrect_answer_1': {
                        'type': 'string',
                        'description': 'The first incorrect answer.'
                    },
                    'incorrect_answer_2': {
                        'type': 'string',
                        'description': 'The second incorrect answer.'
                    },
                    'incorrect_answer_3': {
                        'type': 'string',
                        'description': 'The third incorrect answer.'
                    },
                    'correct_answer': {
                        'type': 'string',
                        'description': 'The correct answer.'
                    },
                },
                'required': ['question', 'incorrect_answer_1', 'incorrect_answer_2', 'incorrect_answer_3', 'correct_answer'],
            },
        }

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.MODEL,
                    messages=messages,
                    functions=[quiz_function],
                    function_call={'name': QUIZ_FUNCTION_NAME},
                )

                arguments = json.loads(response['choices'][0]['message']['function_call']['arguments'])
                break
            except Exception as e:
                print(e)
                continue

        correct_idx = random.randrange(4)
        options = [arguments['incorrect_answer_1'], arguments['incorrect_answer_2'], arguments['incorrect_answer_3']]
        options.insert(correct_idx, arguments['correct_answer'])

        return Question(question=arguments['question'], options=options, answer=correct_idx, page_indices=[page.idx for page in pages])
    

    def _generate_flashcards(self, title: str, pages: Sequence[Page]) -> Flashcards:
        flashcards = []
        batch_size = 1  # Set the batch size to 1 page

        for batch in tqdm(batched(pages, batch_size)):
            # Generate a flashcard for the batch of pages
            flashcard = self._generate_flashcard(batch)
            flashcards.append(flashcard)

        return Flashcards(name='Flashcards', cards=flashcards)
    
    def _generate_flashcard(self, pages: Sequence[Page]) -> Flashcard:
        messages = [
            {
                'role': 'user',
                'content': 'Please generate a flashcard of an important concept from these textbook pages.'
            },
            *[{
                'role': 'user',
                'content': page.text
            } for page in pages]
        ]

        FLASHCARD_FUNCTION_NAME = 'create_flashcard'

        flashcard_function = {
            'name': FLASHCARD_FUNCTION_NAME,
            'description': 'Displays a flashcard',
            'parameters': {
                'type': 'object',
                'properties': {
                    'front': {
                        'type': 'string',
                        'description': 'The front of the flashcard.'
                    },
                    'back': {
                        'type': 'string',
                        'description': 'The back of the flashcard.'
                    },
                },
                'required': ['front', 'back'],
            },
        }

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.MODEL,
                    messages=messages,
                    functions=[flashcard_function],
                    function_call={'name': FLASHCARD_FUNCTION_NAME},
                )

                arguments = json.loads(response['choices'][0]['message']['function_call']['arguments'])
                break
            except Exception as e:
                print(e)
                continue

        flashcard = Flashcard(front=arguments['front'], back=arguments['back'], page_indices=[page.idx for page in pages])

        return flashcard
