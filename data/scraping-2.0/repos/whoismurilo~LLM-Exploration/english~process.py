from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath

    @staticmethod
    def split_pdf(filepath):
        reader = PdfReader(filepath)
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"  # Add newline to separate pages

        if not raw_text.strip():  # Check if raw_text is empty
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_text(raw_text)

    def process_files(self):
        all_texts = []
        for file in self.filepath:
            texts = self.split_pdf(file)
            all_texts.extend(texts)

        # Process each chunk to create a corpus
        corpus = [sentence.lower().split() for text in all_texts for sentence in text.split('\n') if sentence.strip()]
        vocabulary = set(word for sentence in corpus for word in sentence)
        word_to_index = {word: i for i, word in enumerate(vocabulary)}
        vocab_size = len(vocabulary)

        # Create training data
        context_window = 1  # For predicting the next word, a context window of 1 is sufficient
        training_data = []
        for sentence in corpus:
            indices = [word_to_index[word] for word in sentence]
            for i in range(len(indices) - 1):
                input_word_idx = indices[i]
                target_word_idx = indices[i + 1]
                training_data.append((input_word_idx, target_word_idx))

        return word_to_index, vocab_size, training_data
