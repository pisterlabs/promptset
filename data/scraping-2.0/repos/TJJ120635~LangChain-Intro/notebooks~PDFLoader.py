import PyPDF2
import re
from langchain.docstore.document import Document


def PDFLoader(pdf_path: str, max_len: int = 300) -> list:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract PDF content
        pdf_content = ''.join(page.extract_text() for page in pdf_reader.pages)

        # Clean up symbols
        pdf_content = re.sub(r'\n+', '', pdf_content)
        pdf_content = re.sub(r'\s+', ' ', pdf_content)

        # Split into sentences
        sentence_separator_pattern = re.compile('([；。！! \?？]+)')
        sentences = [
            element
            for element in sentence_separator_pattern.split(pdf_content)
            if not sentence_separator_pattern.match(element) and element
        ]

        # Merge sentences into paragraphs
        paragraphs = []
        current_length = 0
        current_paragraph = ""

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_len:
                current_paragraph += sentence
                current_length += sentence_length
            else:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence
                current_length = sentence_length

        paragraphs.append(current_paragraph.strip())
        
        documents = []
        metadata = {"source": pdf_path}
        for para in paragraphs:
            new_doc = Document(page_content=para, metadata=metadata)
            documents.append(new_doc)

    return documents

