import langchain
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter
import os
from supabase import create_client, Client
import openai
from dotenv import dotenv_values

# Download the 'punkt' resource from NLTK
# nltk.download('punkt')
config = dotenv_values(".env") 

supabase: Client = create_client(config['SUPABASE_PROJECT_URL'], config['SUPABASE_API_KEY'])
openai.api_key=config['OPENAI_API_KEYS']

def is_index_page(page_content):
    # print(page_content)
    # Add your logic here to determine if the page is the index page
    # For example, you can check if it contains certain keywords or patterns
    return False  # Return True if it's the index page, False otherwise


def extract_pdf_content(file_path, start, end):
    content = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = end
        start_page = start  # Starting page number for chapter one
        for page_num in range(start_page, num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if not is_index_page(text):  # Skip index page
                content += text
    return content

# def split_text_into_chunks(text, chunk_size):
    # Split text into sentences using NLTK's sent_tokenize
    # sentences = sent_tokenize(text)

    # # Initialize the language chain with spaCy
    # nlp = spacy.load("en_core_web_sm")

    # chunks = []
    # current_chunk = []

    # # Iterate through each sentence
    # for sentence in sentences:
    #     doc = nlp(sentence)

    #     # Process each token in the sentence
    #     for token in doc:
    #         # Check if the token is a punctuation mark or a space
    #         if token.is_punct or token.is_space:
    #             # Add the current chunk to the list of chunks
    #             if current_chunk:
    #                 chunks.append(" ".join(current_chunk))
    #                 current_chunk = []
    #         else:
    #             # Add the token to the current chunk
    #             current_chunk.append(token.text)

    #             # Check if the current chunk has reached the desired size
    #             if len(current_chunk) >= chunk_size:
    #                 chunks.append(" ".join(current_chunk))
    #                 current_chunk = []

    # # Add the last chunk to the list of chunks
    # if current_chunk:
    #     chunks.append(" ".join(current_chunk))

    # return chunks



# file:///C:/Users/HP/LawSchoolEmbedding/trainingDoc/(NLS)%20CRIMINAL%20LITIGATION%20HANDOUT.pdf
# Provide the path to your PDF file
# pdf_file_path = "(NLS) CRIMINAL LITIGATION HANDOUT.pdf"
# extracted_text = extract_pdf_content(pdf_file_path)

# print(extracted_text)
# chunks = split_text_into_chunks(extracted_text, 1000)
# print(len(chunks))
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# text_splitter1 = NLTKTextSplitter(chunk_size=1000)
# pages = text_splitter.split_text(extracted_text)
# pages1 = text_splitter1.split_text(extracted_text)
# # text1 = text_splitter1.create_documents(pages1)
# # print(len(pages))
# # print(len(pages1))
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.create_documents(pages)

# print (texts)
trainingDoc = [
    # {'path': 'trainingDoc/(NLS) CRIMINAL LITIGATION HANDOUT.pdf', 'start': 248, 'end': 263},
    # {'path': 'trainingDoc/(NLS) PROFESSIONAL ETHICS AND SKILLS HANDOUT.pdf', 'start': 97, 'end': 304}, # i stopped here
    # {'path': 'trainingDoc/(NLS) PROPERTY LAW PRACTICE HANDOUT.pdf', 'start': 64, 'end': 284},
    # {'path': 'trainingDoc/3. 2021-2022_Civil Litigation.pdf', 'start': 383, 'end': 412},
    # {'path': 'trainingDoc/2012 - 2018 MCQ.pdf', 'start': 2, 'end': 142},
    # {'path': 'trainingDoc/2016 CIVIL LITIGATION.pdf', 'start': 1, 'end': 27},
    # {'path': 'trainingDoc/2016 CORPORATE LAW PRACTICE-755.pdf', 'start': 1, 'end': 39},
    # {'path': 'trainingDoc/2016 CRIMINAL LITIGATION .pdf', 'start': 1, 'end': 31},
    # {'path': 'trainingDoc/2016 PROFESSIONAL ETHICS AND SKILLS.pdf', 'start': 1, 'end': 24},
    # {'path': 'trainingDoc/2016 PROPERTY LAW PRACTICE.pdf', 'start': 1, 'end': 33},
    # {'path': 'trainingDoc/2018 CIVIL LITIGATION.pdf', 'start': 1, 'end': 35},
    # {'path': 'trainingDoc/2018 CORPORATE LAW PRACTICE.pdf', 'start': 1, 'end': 35},
    # {'path': 'trainingDoc/2018 CRIMINAL LITIGATION.pdf', 'start': 1, 'end': 24},
    # {'path': 'trainingDoc/2018 PROFESSIONAL ETHICS AND SKILLS.pdf', 'start': 1, 'end': 24},
    # {'path': 'trainingDoc/2018 PROPERTY LAW PRACTICE .pdf', 'start': 1, 'end': 35},
    # {'path': 'trainingDoc/2019 CIVIL LITIGATION.pdf', 'start': 1, 'end': 41},
    # {'path': 'trainingDoc/2019 CORPORATE LAW PRACTICE.pdf', 'start': 1, 'end': 30},
    # {'path': 'trainingDoc/2019 CRIMINAL LITIGATION.pdf', 'start': 1, 'end': 27},
    # {'path': 'trainingDoc/2019 PROPERTY LAW PRACTICE.pdf', 'start': 1, 'end': 32},
    # {'path': 'trainingDoc/2020 CIVIL LITIGATION MCQ QUESTIONS.pdf', 'start': 1, 'end': 9},
    # {'path': 'trainingDoc/2020 CIVIL LITIGATION.pdf', 'start': 1, 'end': 37},
    # {'path': 'trainingDoc/2020 CORPORATE LAW PRACTICE MCQ QUESTIONS.pdf', 'start': 1, 'end': 7},
    # {'path': 'trainingDoc/2020 CORPORATE LAW PRACTICE.pdf', 'start': 1, 'end': 20},
    # {'path': 'trainingDoc/2020 CRIMINAL LITIGATION MCQ QUESTIONS.pdf', 'start': 1, 'end': 7},
    # {'path': 'trainingDoc/2020 CRIMINAL LITIGATION PQ&A.pdf', 'start': 1, 'end': 31},
    # {'path': 'trainingDoc/2020 PROFESSIONAL ETHICS & SKILLS.pdf', 'start': 1, 'end': 25},
    # {'path': 'trainingDoc/2020 PROPERTY LAW PRACTICE.pdf', 'start': 1, 'end': 37},
    # {'path': 'trainingDoc/CIVIL LITIGATION HANDBOOK 2022.pdf', 'start': 169, 'end': 316},
    # {'path': 'trainingDoc/CIVIL LITIGATION-1.pdf', 'start': 180, 'end': 188},
    # {'path': 'trainingDoc/COOPORATE LAW PRACTICES HANDBOOK 2022.pdf', 'start': 44, 'end': 288},
    # {'path': 'trainingDoc/Corporate Law - Killi Nancwat.pdf', 'start': 2, 'end': 290},
    # {'path': 'trainingDoc/Criminal Litigation - Killi Nancwat.pdf', 'start': 143, 'end': 214},
    # {'path': 'trainingDoc/DRAFTS ON CIVIL LITIGATION.pdf', 'start': 17, 'end': 144},
    # {'path': 'trainingDoc/DRAFTS ON CORPORATE LAW PRACTICE.pdf', 'start': 2, 'end': 97},
    # {'path': 'trainingDoc/DRAFTS ON CRIMINAL LITIGATION.pdf', 'start': 2, 'end': 64},
    # {'path': 'trainingDoc/DRAFTS ON PROFESSIONAL ETHICS.pdf', 'start': 2, 'end': 74},
    # {'path': 'trainingDoc/DRAFTS ON PROPERTY LAW PRACTICE.pdf', 'start': 2, 'end': 76},
    # {'path': 'trainingDoc/Ethics and Skill  KILL NANCWAT.pdf', 'start': 2, 'end': 198},
    # {'path': 'trainingDoc/MCQ 2019.pdf', 'start': 1, 'end': 39},
    # {'path': 'trainingDoc/MCQs, Civil Litigation.pdf', 'start': 2, 'end': 86}, # already embedded
    # {'path': 'trainingDoc/MCQs, Corporate Law Practice.pdf', 'start': 2, 'end': 89},
    # {'path': 'trainingDoc/MCQs, Criminal Litigation.pdf', 'start': 2, 'end': 100},
    # {'path': 'trainingDoc/MCQs, Professional Ethics.pdf', 'start': 2, 'end': 70},
    # {'path': 'trainingDoc/MCQs, Property Law Practice-1.pdf', 'start': 46, 'end': 72},
    # {'path': 'trainingDoc/Property Law - Killi Nancwat.pdf', 'start': 50, 'end': 213},
    {'path': 'trainingDoc/PROPERTY LAW.pdf', 'start': 41, 'end': 114},
    # {'path': 'trainingDoc/week 2- Introduction to Criminal Litigation.pdf', 'start': 2, 'end': 114},
]

def get_embedding(text):
    text = text.replace("\n", " ")
    embeddingResponse = openai.Embedding.create(input=[text], model='text-embedding-ada-002')['data'][0]['embedding']
    supabase.table("documents").insert({"content": text, 'embedding': embeddingResponse}).execute()

for doc in trainingDoc:
    path = doc['path']
    start = doc['start']
    end = doc['end']
    extracted_text = extract_pdf_content(path, start, end)
    text_splitter1 = NLTKTextSplitter(chunk_size=1000)
    pages1 = text_splitter1.split_text(extracted_text)
    print(pages1)
    print(len(pages1))
    for page in pages1:
        embeddingData = get_embedding(page)





print('Hello')
