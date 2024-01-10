import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re

my_activeloop_org_id = "abduljaweed"
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path= 'hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}'

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)



def get_documentation_urls():
    # List of relative URLs for Hugging Face documentation pages, commented a lot of these because it would take too long to scrape all of them
    
    return [
        '/docs/huggingface_hub/guides/overview',
		'/docs/huggingface_hub/guides/download',
	    '/docs/huggingface_hub/guides/upload',
	    '/docs/huggingface_hub/guides/hf_file_system',
	    '/docs/huggingface_hub/guides/repository',
	    '/docs/huggingface_hub/guides/search',
	    # You may add additional URLs here or replace all of them
    ]
    

def construct_full_url(base_url, relative_url):
    # Construct the full URL by appending the relative URL to the base URL
    return base_url, relative_url


def scrape_page_content(url):
    # Send a GET request to the URL and parse the HTML response using BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract the desired content from the page (in this case, the body text)
    text=soup.body.text.strip()
    # Remove non-ASCII characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def scrape_all_content(base_url, relative_urls, filename):
    # Loop through the list of URLs, scrape content and add it to the content list
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    # Write the scraped content to a file
    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)
    
    return content


# Define a function to load documents from a file

def load_docs(root_dir, filename):
    # Create an empty list to hold the documents
    docs = []
    
    try:
        # Load the file using the TextLoader class and UTF-8 encoding
        loader = TextLoader(os.path.join(
            root_dir, filename
        ),encoding='utf-8')
        # Split the loaded file into separate documents and add them to the list of documents
        docs.extend(loader.load_and_split())
    except Exception as e:
         # If an error occurs during loading, ignore it and return an empty list of documents
        pass
     # Return the list of documents
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    return text_splitter.split_documents(docs)


# Define the main function
def main():
    base_url = 'https://huggingface.co'
    # Set the name of the file to which the scraped content will be saved
    filename='content.txt'
    # Set the root directory where the content file will be saved
    root_dir ='./'
    relative_urls = get_documentation_urls()
    # Scrape all the content from the relative URLs and save it to the content file
    content = scrape_all_content(base_url, relative_urls,filename)
    # Load the content from the file
    docs = load_docs(root_dir,filename)
    # Split the content into individual documents
    texts = split_docs(docs)
    # Create a DeepLake database with the given dataset path and embedding function
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # Add the individual documents to the database
    db.add_documents(texts)
    # Clean up by deleting the content file
    os.remove(filename)

# Call the main function if this script is being run as the main program
if __name__ == '__main__':
    main()