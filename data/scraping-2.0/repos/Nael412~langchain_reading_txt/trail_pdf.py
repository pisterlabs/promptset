
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
import bs4

# pdf_path = '/Users/naelshaker/Work/pdf to txt langchain/langchain_reading_txt/Hooked How to Build Habit-Forming Products • Supplemental Workbook (Nir Eyal, Ryan Hoover) (z-lib.org).pdf'
# epub_path = '/Users/naelshaker/Work/pdf to txt langchain/langchain_reading_txt/Inspired_ How to Create Tech Products Customers Love by Marty Cagan.epub'
epub_path = 'No Rules Rules.epub'
output_file = 'output.txt'

# # Read the PDF
# reader = PdfReader(pdf_path)

# # Extract text from each page
# text = ''
# for page in reader.pages:
#     text += page.extract_text()
    

# Open the EPUB file
book = epub.read_epub(epub_path)

# Extract text from each chapter to text 
text = ''
for item in book.get_items():
    if isinstance(item, epub.EpubHtml):
        soup = bs4.BeautifulSoup(item.get_content(), 'html.parser')
        text += soup.get_text()

# Write the extracted text to a text file
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(text)

print(f"Output written to {output_file}.")

#loading .env to get api key 
load_dotenv()

#making embeddings for the text file
embeddings = OpenAIEmbeddings()

#get text from text file 
loader = TextLoader('output.txt')
#get it from directory
# loader = DirectoryLoader('news', glob="**/*.txt")
#loading the whatever is in loader
documents = loader.load()

#This split the text to chunks and specify if I want any overlap
text_splitter = CharacterTextSplitter(chunk_size= 2500, chunk_overlap = 0)
texts = text_splitter.split_documents(documents)


#store embeddings in chroma as vector store
vecstore = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    retriever = vecstore.as_retriever()
)

#this function takes the question and print with the answer running on 
def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

#Queries for the news directory
# query("What are the effects of legistlations surrounding emissions on the Austrailian coal market?")
# query("What are China's plan with renewable energy?")
# query("Is there an export ban on coal in indonesia?")

#Queries for the inspired book 
query("How do they manage teams at Netflix?")
query("What does the company think about vacation policy?")
query("How to create autonomous teams according to the book?")

#printing length of  documents
# print (len(documents))



# text = "Algorithma is a data science school based in indonesia and super type is a data science consultancy"
# doc_embeddings = embeddings.embed_documents([text])

# print(doc_embeddings)s

