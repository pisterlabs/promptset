from PyPDF2 import PdfReader
import re
import pandas as pd
import nltk
from textblob import TextBlob
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import dotenv
import time
import json

from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
import textwrap


dotenv.load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load spacy's NER model
nlp = spacy.load('en_core_web_sm')

patterns_to_remove = [
    r'\(see Figure \d+\)',  # More specific pattern to remove references to figures
    r'\(see Table \d+\)',   # More specific pattern to remove references to tables
    # Add more specific patterns as needed
]

def rename_pdfs(folder_path):
    files = os.listdir(folder_path)
    
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    pdf_files.sort()
    for i, file in enumerate(pdf_files):
        old_file = os.path.join(folder_path, file)
        new_file = os.path.join(folder_path, f"{i}.pdf")
        try:
            
            os.rename(old_file, new_file)
            
        except Exception as e:
            print(e)
            

# Function to clean text using regular expressions
def clean_text(text):
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Remove any standalone mathematical symbols and numbers not part of words
    text = re.sub(r'\b[\d.]+\b', '', text)
    # Remove any additional unwanted characters, keep hyphens and apostrophes within words
    text = re.sub(r'(?<!\w)[-]', ' ', text)
    text = re.sub(r'(?<!\w)[^\w\s\'-]+', '', text)
    text = re.sub(r'(?<=\w)[^\w\s\'-]+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\[a-z]*', ' ', text)
    return text


def find_nearest_section(text, keywords):
    nearest_index = len(text)
    for keyword in keywords:
        index = text.lower().find(keyword)
        if 0 <= index < nearest_index:
            nearest_index = index
    return nearest_index

def extract_conclusion(text):
    conclusion_index = find_nearest_section(text, ['conclusion'])
    conclusion = text[conclusion_index:conclusion_index+1500].strip()
    ref_index = conclusion.lower().find("references")
    conclusion = conclusion[:ref_index]
    return conclusion
    
    
def text_processing(text: str) -> str:
    # Lowercasing
    text = text.lower()

    # Removing HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace URLs and email addresses with a space
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', ' ', text)

    # Carefully replace or remove non-standard characters
    # This regex will replace non-alphanumeric characters that are not within a word with a space
    text = re.sub(r'(?<=\W)\W+|\W+(?=\W)', ' ', text)

    # Tokenization and further cleaning
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]

    return ' '.join(cleaned_words)
    
def get_summary(file_path):
    '''
    This function takes file path as input and leveraging the open ai function to generate a summary which we will use as our output column
    '''
    # Load your language model (LLM)
    llm = OpenAI(temperature=0)

    max_token_count=4000
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=3000)
    docs = text_splitter.split_documents(doc)[:4]
    
    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce")


    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    return str(output_summary).strip()
    
if __name__=='__main__':
    # rename_pdfs("research_papers/")
    tot_papers = len(os.listdir("research_papers/"))
    data_dict = {'text': [], 'summary': []}
    for i in range(0, tot_papers):## Loop through all the research papers
        reader = PdfReader(f'./research_papers/{i}.pdf')## read the paper
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        
        text = clean_text(text) ## clean the paper text
        try:
            summary = get_summary(f'./research_papers/{i}.pdf') ## generate a summary 
        except Exception as e:
            summary = ''
        
        # print(summary)
        print(f"{i}: {len(summary)}")
        
        data_dict['text'].append(text)
        data_dict['summary'].append(summary)
        
        ## storing this to retain the information if the above code runs into some error, so that we have result until the error generated
        with open("summary1.json", "w") as outfile: 
            json.dump(data_dict, outfile)
    
    ## Finally create our dataset which is ready to be trained
    df = pd.DataFrame(data=data_dict)
    df['processed_text'] = df['text'].apply(text_processing)
    
    df.to_csv('final.csv', index=False)
    
    print(df.shape)
    

#28, 40, 42, 69