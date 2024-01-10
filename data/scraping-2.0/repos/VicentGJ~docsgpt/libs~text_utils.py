import io
import os
import tempfile
import textract
import ai21
from PyPDF2 import PdfReader
from docx import Document
from googletrans import Translator
from langchain.text_splitter import CharacterTextSplitter


def get_text_from_file(file):
    """
    Extracts text from a file.

    :param file: The file to extract text from.
    :type file: File object
    :return: The extracted text.
    :rtype: str
    """
    # If the file is a PDF, use the PyPDF2 library to extract the text
    if file.type == "application/pdf":
        text = get_pdf_text(file)
    else:
        # Other formats allowed by textract: doc, docx, eml, epub, html, json, rtf, txt, odt
        # TODO Test all the textract formats

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            # Write the uploaded file data to the temporary file
            temp.write(file.read())
            # Get the path of the temporary file
            temp_path = temp.name

        # Extract the text from the temporary file
        text = extract_text(temp_path)

        # Delete the temporary file
        os.remove(temp_path)

    return text


def get_pdf_text(pdf):
    """
    Extracts text from a PDF file.

    :param pdf: The PDF file to extract text from.
    :type pdf: File object
    :return: The extracted text.
    :rtype: str
    """
    # Initialize an empty string to store the extracted text
    text = ""

    # Create a PDF reader object using the PyPDF2 library
    pdf_reader = PdfReader(pdf)

    # Iterate over the pages in the PDF file
    for page in pdf_reader.pages:
        # Extract the text from the current page and append it to the text string
        text += page.extract_text()

    # Return the extracted text
    return text


def get_docx_text(doc):
    """
    Extracts text from a docx file.

    :param doc: The docx file to extract text from.
    :type doc: File object
    :return: The extracted text.
    :rtype: str
    """
    # Load the docx file data into a stream
    doc_stream = io.BytesIO(doc.read())

    # Create a Document object from the stream using the python-docx library
    document = Document(doc_stream)

    # Extract the text from the document by iterating over its paragraphs
    text = '\n'.join([para.text for para in document.paragraphs])

    # Return the extracted text
    return text


def get_text_chunks(text, chunk_size, chunk_overlap):
    """
    Splits text into chunks of a specified size with a specified overlap.

    :param text: The text to split into chunks.
    :type text: str
    :param chunk_size: The size of each chunk.
    :type chunk_size: int
    :param chunk_overlap: The overlap between chunks.
    :type chunk_overlap: int
    :return: A list of text chunks.
    :rtype: list of str
    """
    # Create a CharacterTextSplitter object using the specified parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Split the text into chunks using the splitter object
    return text_splitter.split_text(text)


def get_text_segments(source: str, sourceType: str) -> list[str]:
    """
    Splits text into segments using the AI21 Studio API.

    :param source: The text to split into segments.
    :type source: str
    :param sourceType: The type of the source. Must be "TEXT" or "URL".
    :type sourceType: str
    :return: A list of text segments.
    :rtype: list of str
    """
    # Initialize an empty list to store the segments
    segments = []

    # Initialize a list containing the source text
    texts: list[str] = [source]

    # Initialize a counter variable
    i = 0

    # If the source type is "TEXT" and there are still texts to process
    while sourceType == "TEXT" and i < len(texts):
        # If the current text is longer than 100000 characters
        if len(texts[i]) > 100000:
            # Split the text in half
            middle_index = int(len(texts[i]) / 2)
            first_half = texts[i][:middle_index]
            second_half = texts[i][middle_index:]

            # Remove the current text from the list and insert its two halves
            index = texts.index(texts[i])
            texts.remove(texts[i])
            texts.insert(index, second_half)
            texts.insert(index, first_half)
        else:
            # Move on to the next text
            i += 1

    # Iterate over the texts
    for text in texts:
        # Use the AI21 Studio API to split the text into segments
        response = ai21.Segmentation.execute(
            source=text,
            sourceType=sourceType
        )

        # Extract the segments from the response
        segments_list = response["segments"]

        # Append each segment to the segments list
        for dict in segments_list:
            segments.append(dict['segmentText'])

    # Return the list of segments
    return segments


def get_text_from_path(path: str) -> str:
    """
    Extracts text from a file at a specified path.

    :param path: The path of the file to extract text from.
    :type path: str
    :return: The extracted text.
    :rtype: str
    """
    # If the file is a PDF
    if path.endswith(".pdf"):
        # Use the get_pdf_text function to extract the text
        text = get_pdf_text(path)
    else:
        # Use the textract library to extract the text
        text = extract_text(path)

    # Return the extracted text
    return text


def extract_text(file_path: str) -> str:
    """
    Extracts text from a file at the given path.
    """
    # Determine the file extension
    extension = file_path.split('.')[-1]
    # Extract the text from the temporary file using textract
    text = textract.process(file_path, extension=extension).decode('utf-8')

    # Return the extracted text
    return text


def translate_text(text: str, dest: str) -> str:
    """
    Translates text into a specified language using the Google Translate API.

    :param text: The text to translate.
    :type text: str
    :param dest: The language to translate the text into.
    :type dest: str
    :return: The translated text.
    :rtype: str
    """
    # Create a Translator object using the googletrans library
    trans = Translator()

    # Use the Translator object to translate the text
    resp = trans.translate(text=text, dest=dest)

    # Return the translated text
    return resp.text


def detect_text_language(text: str) -> str:
    """
    Detects the language of a text using the Google Translate API.

    :param text: The text to detect the language of.
    :type text: str
    :return: The detected language.
    :rtype: str
    """
    # Create a Translator object using the googletrans library
    trans = Translator()

    # Use the Translator object to detect the language of the text
    return trans.detect(text).lang
