#This is used for the set up and intiallising the modules and importing them
import tkinter as tk
from tkinter import Toplevel
from tkinter import messagebox
from PIL import Image, ImageTk
import PyPDF2
import spacy
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter 

# Dictionary to store keywords from different pages

page_keywords = {}

# Class representing the QueryApp
class QueryApp:
    def __init__(self):
        
        # Initialize the GUI
        self.init()
        self.question = ""

    def init(self):
        # Create the main GUI window
        self.root = tk.Tk()
        self.root.title("Query Form")
        
         # Load and display the logo
        self.logo = Image.open("iisc_logo.jpg")
        self.logo = self.logo.resize((200, 200))
        self.logo = ImageTk.PhotoImage(self.logo)
        
        # Create the main page elements
        self.create_main_page()
        self.solution_pages = []
        self.initialize_text_processing()


    def create_main_page(self):
        # GUI code for creating the main page
        # Configure the background color
        self.root.configure(bg="lightgray")
        
        # Display the logo
        logo_label = tk.Label(self.root, image=self.logo)
        logo_label.pack(pady=10)

        # Display a welcome message

        welcome_label = tk.Label(self.root, text="Welcome to the Query Form!", font=("Helvetica", 24, "bold"), fg="purple", bg="lightgray")
        welcome_label.pack(pady=20)
        
        
        # Label for entering the query
        query_label = tk.Label(self.root, text="Please enter your query:", font=("Helvetica", 14), bg="lightgray")
        query_label.pack()

        # Text entry widget for the query

        self.query_entry = tk.Text(self.root, height=5, width=60, font=("Times Roman", 12))
        self.query_entry.pack(pady=10, padx=10)

        # Button to submit the query
        submit_button = tk.Button(self.root, text="Submit Query", command=self.submit_query, font=("Helvetica", 14, "bold"), bg="green", fg="white")
        submit_button.pack()

    def submit_query(self):
        # GUI code for processing the submitted query
        # Get the query text from the text entry widget
        query = self.query_entry.get("1.0", "end-1c")
        self.show_solution_page(query)

    def show_solution_page(self, query):
        # GUI code for displaying the solution page
        # Create a new window for the solution page

        solution_root = Toplevel(self.root)  # Use Toplevel for additional windows
        solution_root.title("Solution Page")
        
        # Label for the solution
        solution_label = tk.Label(solution_root, text="Solution for your query:", font=("Helvetica", 18, "bold"), fg="purple")
        solution_label.pack(pady=20)
        
        # Text widget to display the solution
        self.solution_text = tk.Text(solution_root, height=10, width=70, font=("Helvetica", 12), wrap="word", spacing1=5, spacing2=3, spacing3=5)
        self.solution_text.tag_configure("solution_tag", font=("Helvetica", 12), background="lightyellow")
        self.solution_text.pack(pady=10)
        

        # Button to go back to the query form
        back_button = tk.Button(solution_root, text="Back to Query Form", command=solution_root.destroy, font=("Helvetica", 12), bg="red", fg="white")
        back_button.pack()

        
         # Insert the query and initial solution information
        self.solution_text.insert("1.0", f"Query: {query}\n\n")
        self.solution_text.insert("end", "Solution:", "solution_tag")

         # Further query entry and button

        further_query_label = tk.Label(solution_root, text="Have another query?", font=("Helvetica", 12))
        further_query_label.pack()

        
        self.further_query_entry = tk.Text(solution_root, height=2, width=50, font=("Helvetica", 12))
        self.further_query_entry.pack(pady=10)

        further_query_button = tk.Button(solution_root, text="Submit Further Query", command=self.submit_further_query, font=("Helvetica", 12, "bold"), bg="orange", fg="white")
        further_query_button.pack()

        self.solution_pages.append(solution_root)

         # Add the solution page to the list for further reference

    def submit_further_query(self):

        #GUI code for submitting a further query
        # Get the further query text
        further_query = self.further_query_entry.get("1.0", "end-1c")
        # Update the solution text with the user's further query
        self.solution_text.insert("end", f"User Query: {further_query}\n", "solution_tag")
        # Clear the further query entry
        self.further_query_entry.delete("1.0", "end")
        # Display the solution page for the further query
        self.show_solution_page(further_query)



    def run(self):
        # Start the main GUI event loop
        self.root.mainloop()

    def run(self):
        self.root.mainloop()

    #Below is used for the asking question by the user
    def get_user_question(self):             
         return input("Ask a question (or type 'exit' to quit): ")  # Modify as needed
        # question = input("your question")
    
    #Actual processing of the question and answering
    def initialize_text_processing(self):
        pdf_path = "AltoK10.pdf" #set this pat as acoording to your pdf path
        self.page_keywords = extract_keywords_from_page(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")

        self.question = self.get_user_question()  # Set the value of 'question'

        db, llm = self.load_text_processing_resources()

        retriever = db.as_retriever(search_kwargs={"k": 3})
        retrieval_qa = RetrievalQA(llm)

        res = retrieval_qa.qa(self.question)
        print(res["result"])

    #This is used to load the pdf
    def load_text_processing_resources(self):
        loader = PyPDFLoader("Altok10.pdf")#Replace this according to your your pdf if neccesary
        documents = loader.load_and_split()

        #This is used to keep recurisve text 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        texts = text_splitter.split_documents(documents)

        #This is used to load the hugging-face with the model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        #This is used to create the data-base
        db = Chroma.from_documents(texts, embeddings, persist_directory="db")

        #Initiallising the llm model
        llm = GPT4All(
            model="./ggml-gpt4all-j-v1.3-groovy.bin",
            backend="gptj",
            verbose=False
        )

        return db, llm 
    
#Intiallising the page ranges with chapter names    
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

#Extracting the keywords
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

#Extracting the chapter names based on keywords
def get_chapter_for_keyword(keyword):
    for page_number, keywords in page_keywords.items():
        if keyword in keywords:
            for page_range, content in page_ranges.items():
                start_page, end_page = page_range
                if start_page <= page_number <= end_page:
                    return content
    return "Keyword not found in the document"

# Create an instance of the QueryApp class and run the application
if __name__ == "__main__":
    app = QueryApp()
    app.run()
