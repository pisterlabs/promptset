import heapq
import os
import warnings
import nltk

from langchain.document_loaders import TextLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from text import AutoTranslator

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning)  # Hide download warnings
    nltk.download('stopwords', quiet=True)

# Define constants for folder path and language dictionary
FOLDER_PATH = "docs"
LANGUAGE_DICT = {"id": "indonesian", "en": "english"}

class Document:
    """A class to represent a document with its name, content, and score."""

    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.score = 0

    def __repr__(self):
        return f"Document(name={self.name}, score={self.score})"

class DocumentSearchBackend:
    def __init__(self):
        self.translator = AutoTranslator()
        self.document_language = None

    def detect_language(self, text: str):
        """Detect the language of a text using the translator."""
        detected_language = self.translator.detect_language(text, sanitize=True)
        return LANGUAGE_DICT.get(detected_language)

    def process_documents(self):
        """Process the documents in the folder and return a list of Document objects."""
        documents = []

        file_list = self.get_text_files()

        for file_name in tqdm(file_list, desc='Processing documents'):
            texts = self.load_texts(file_name)

            for text in texts:
                document = Document(file_name, text.page_content)
                document.content = self.tokenize_without_stopwords(document.content)
                documents.append(document)

        return documents

    def get_text_files(self):
        """Get a list of text files in the folder."""
        file_list = [file_name for file_name in os.listdir(FOLDER_PATH) if file_name.endswith('.txt')]
        return file_list

    def load_texts(self, file_name: str):
        """Load texts from a file using TextLoader."""
        text_loader = TextLoader(os.path.join(FOLDER_PATH, file_name), encoding="utf-8")
        texts = text_loader.load()
        return texts

    def detect_document_language(self, documents: list):
        """Detect the language of the documents using the first non-empty text."""
        for document in documents:
            word_count = len(document.content)

            if 10 < word_count < 50:
                self.document_language = self.detect_language(" ".join(document.content))
                break

        if self.document_language is None:
            # Use the first document as a fallback
            self.document_language = self.detect_language(" ".join(documents[0].content))

    def tokenize_without_stopwords(self, text: str):
        """Tokenize a text without stopwords using nltk."""
        language = self.detect_language(text)
        words = nltk.word_tokenize(text)
        language_stopwords = stopwords.words(language)
        return [word for word in words if word not in language_stopwords]

    def get_keywords(self, question: str) -> list:
        """Get keywords from a question by removing stopwords and punctuation."""
        language = self.detect_language(question)
        stopwords_language = set(stopwords.words(language))
        custom_stopwords = {"?", "!", ".", ","}
        stopwords_language.update(custom_stopwords)

        tokens = word_tokenize(question)
        keywords = [token for token in tokens if token not in stopwords_language]

        return list(set(keywords))

    def find_top_documents(self, keywords: list, documents: list) -> list:
        """Find the top documents that match the keywords using BM25Okapi."""
        # Translate the keywords to the document language
        translated_keywords = self.translator.auto_translate_keywords(keywords, self.document_language)

        # Create a BM25Okapi object with the document contents
        bm25 = BM25Okapi([document.content for document in documents])

        # Get the scores for each document
        doc_scores = bm25.get_scores(translated_keywords)

        # Assign the scores to the document objects
        for document, score in zip(documents, doc_scores):
            document.score = score

        # Find the top 3 documents with the highest scores
        top_documents = heapq.nlargest(3, documents, key=lambda x: x.score)

        # Return the top documents as a list
        return top_documents

    def search_documents(self, question: str) -> list:
        """Search the documents for a question and return a list of top documents with their locations and scores."""

        # Get the keywords from the question
        keywords = self.get_keywords(question)

        # Process the documents and create Document objects
        documents = self.process_documents()

        # Detect the document language
        self.detect_document_language(documents)

        # Find the top documents that match the keywords
        top_documents = self.find_top_documents(keywords, documents)

        # Prepare the result list with document names, locations, and scores
        result = []
        for document in top_documents:
            result.append({
                'name': document.name,
                'location': os.path.join(FOLDER_PATH, document.name),
                'score': document.score
            })

        return result[:3]  # Return only the top 3 documents

