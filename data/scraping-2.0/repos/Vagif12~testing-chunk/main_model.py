def detect_document_type(document_path):
    
    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf"
        
    elif(guess_file.extension.lower() in image_types):
        file_type = "image"
        
    else:
        file_type = "unkown"
        
    return file_type

from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader

def extract_file_content(file_path):
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
        
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    
    return documents_content


research_paper_path = "./data/transformer_paper.pdf"
article_information_path = "./data/zoumana_article_information.png"

print(f"Research Paper Type: {detect_document_type(research_paper_path)}")
print(f"Article Information Document Type: {detect_document_type(article_information_path)}")

research_paper_content = extract_file_content(research_paper_path)
article_information_content = extract_file_content(article_information_path)

nb_characters = 400

print(f"First {nb_characters} Characters of the Paper: \n{research_paper_content[:nb_characters]} ...")
print("---"*5)
print(f"First {nb_characters} Characters of Article Information Document :\n {research_paper_content[:nb_characters]} ...")

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

research_paper_chunks = text_splitter.split_text(research_paper_content)
article_information_chunks = text_splitter.split_text(article_information_content)

print(f"# Chunks in Research Paper: {len(research_paper_chunks)}")
print(f"# Chunks in Article Document: {len(article_information_chunks)}")

from langchain.embeddings.openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "<YOUR_KEY>"
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import FAISS

def get_doc_search(text_splitter):
    
    return FAISS.from_texts(text_splitter, embeddings)

doc_search_paper = get_doc_search(research_paper_chunks)
print(doc_search_paper)


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)

def chat_with_file(file_path, query):
    
    file_content = extract_file_content(file_path)
    text_splitter = text_splitter.split_text(file_content)
    
    document_search = get_doc_search(text_splitter)
    documents = document_search.similarity_search(query)
    
    results = chain({
                        "input_documents":documents, 
                        "question": query
                    }, 
                    return_only_outputs=True)
    answers = results['intermediate_steps'][0]
    
    return answers