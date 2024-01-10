#Import Statements
#PDF Processing Setup
import PyPDF2
import spacy
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path

#Text Processing and NLP Setup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

#Keyword Extraction Function (extract_keywords)
def extract_keywords(text, num_keywords=5):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    keyword_counter = Counter(filtered_tokens)
    keywords = [keyword for keyword, _ in keyword_counter.most_common(num_keywords)]
    return keywords

#PDF Document Loading and Splitting
loader = PyPDFLoader("AltoK10.pdf")
documents = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64
)
texts = text_splitter.split_documents(documents)

#Text Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector Store Creation
db = Chroma.from_documents(texts, embeddings, persist_directory="db")

#LLM (Language Model) Initialization
llm = GPT4All(
    model="./ggml-gpt4all-j-v1.3-groovy.bin",
    backend="gptj",
    verbose=False
)

#Page Ranges and Chapter Information
page_ranges = {
    (17, 18): "chapter 1 : ['FUEL RECOMMENDATION']",
    (19, 62): "Chapter 2 : ['BEFORE DRIVING']",
    (63, 81): "Chapter 3: ['OPERATING YOUR VEHICLE']",
    (83, 91): "Chapter 4:['DRIVING TIPS']",
    (93, 131): "Chapter 5:['OTHER CONTROLS AND EQUIPMENT']",
    (133, 136): "Chapter 6:['VEHICLE LOADING AND TOWING']",
    (137, 168): "Chapter 7:['INSPECTION AND MAINTENANCE']",
    (169, 177): "Chapter 8:['EMERGENCY SERVICE']",
    (179, 184): "Chapter 9:['APPEARANCE CARE']",
    (185, 186): "Chapter 10: ['GENERAL INFORMATION']",
    (187, 190): "Chapter 11:['SPECIFICATIONS']",
}

#Keyword Extraction from PDF Pages (extract_keywords_from_page)
def extract_keywords_from_page(pdf_path):
    extracted_keywords_per_page = {}  # Dictionary to store keywords for each page

    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            extracted_text = page.extract_text()

            # Load spaCy NLP model
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(extracted_text)

            # Extract unique keywords (nouns and proper nouns)
            extracted_keywords = set()
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    extracted_keywords.add(token.text)
            
            # Store keywords for the current page
            extracted_keywords_per_page[page_number + 1] = extracted_keywords

    return extracted_keywords_per_page


#Finding Chapters for Keywords (get_chapter_for_keyword)
def get_chapter_for_keyword(keyword):
    for page_number, keywords in page_keywords.items():
        if keyword in keywords:
            for page_range, content in page_ranges.items():
                start_page, end_page = page_range
                if start_page <= page_number <= end_page:
                    return content
    return "Keyword not found in the document"

#Answering User Questions (answer_question)
def answer_question(question):
    question_keywords = set(token.text for token in nlp(question) if token.pos_ in ["NOUN", "PROPN"])
    relevant_chapters = []
    for keyword in question_keywords:
        chapter = get_chapter_for_keyword(keyword)
        if chapter != "Keyword not found in the document":
            relevant_chapters.append(chapter)
    
    if not relevant_chapters:
        return "I couldn't find relevant information for your question."

    return f"Based on your question, the relevant chapters are: {', '.join(relevant_chapters)}"

#User Interaction (get_user_question):
def get_user_question():
    return input("Ask a question (or type 'exit' to quit): ")

#PDF Path and Initial NLP Setup
pdf_path = "AltoK10.pdf"
page_keywords = extract_keywords_from_page(pdf_path)
nlp = spacy.load("en_core_web_sm")

#Infinite Loop for User Interaction
while True:
    question = get_user_question()
    if question.lower() == "exit":
        print("Exiting the question-answering system.")
        break
    answer = answer_question(question)
    print(f"Answer: {answer}\n")

#RetrievalQA Initialization (qa = RetrievalQA.from_chain_type(...))
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)

res = qa(question)     #RetrievalQA Query
print(res["result"])   #Printing the Retrieval Results