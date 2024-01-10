import os
from io import StringIO

import fitz
import openai
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

load_dotenv()
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


def read_pdf(filename):
    context = ""

    # Open the PDF file
    with fitz.open(filename) as pdf_file:
        # Get the number of pages in the PDF file
        num_pages = pdf_file.page_count

        # Loop through each page in the PDF file
        for page_num in range(num_pages):
            # Get the current page
            page = pdf_file[page_num]

            # Get the text from the current page
            page_text = page.get_text().replace("\n", "")

            # Append the text to context
            context += page_text
    return context


def split_text(text, chunk_size=5000):
    """
    Splits the given text into chunks of approximately the specified chunk size.

    Args:
    text (str): The text to split.

    chunk_size (int): The desired size of each chunk (in characters).

    Returns:
    List[str]: A list of chunks, each of approximately the specified chunk size.
    """

    chunks = []
    current_chunk = StringIO()
    current_size = 0
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence_size = len(sentence)
        if sentence_size > chunk_size:
            while sentence_size > chunk_size:
                chunk = sentence[:chunk_size]
                chunks.append(chunk)
                sentence = sentence[chunk_size:]
                sentence_size -= chunk_size
                current_chunk = StringIO()
                current_size = 0
        if current_size + sentence_size < chunk_size:
            current_chunk.write(sentence)
            current_size += sentence_size
        else:
            chunks.append(current_chunk.getvalue())
            current_chunk = StringIO()
            current_chunk.write(sentence)
            current_size = sentence_size
    if current_chunk:
        chunks.append(current_chunk.getvalue())
    return chunks


filename = os.path.join(UPLOAD_FOLDER, "filename.pdf")
document = read_pdf(filename)
chunks = split_text(document)

text = document


def gpt3_completion(content):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": content},
        ],
    )["choices"][0]["message"]["content"]


def ask_question_to_pdf(question):
    return gpt3_completion(
        text
        + " D'après le texte précédent, réponds à la question suivantes: "
        + question
        + " Si ce n'est pas une question demande à l'élève de te poser une question. "
    )


def question():
    return gpt3_completion(text + " Pose moi une question sur le texte précédent.")


def correct_answer(answer, question):
    return gpt3_completion(
        text
        + " La question était: "
        + question
        + ". La réponse de l'élève est: "
        + answer
        + ". Corrige cette réponse si elle est fausse."
    )


def generate_QCM(n):
    return gpt3_completion(
        text
        + "Génère un QCM de exactement "
        + str(n)
        + " question(s) où chaque question possède 3 réponses. Il doit n'y avoir qu'une bonne réponse parmi les trois placéealéatoirement."
        + "Mets le tout dans un format JSON où correct_answer représente l'index de la bonne réponse compris entre 0 et 2 avec le format 'questions': [ { 'question': ... , 'answers': [ reponse a, reponse b, reponse c ], 'correct_answer': ... }, ... ]"
    )


def generate_QCM_answer(qcm, answers):
    return gpt3_completion(
        text
        + "Voici un qcm sur le cours :"
        + qcm
        + ". Voici les réponses de l'étudiant:"
        + answers
        + ". Corrige les réponses de l'étudiant."
    )
