import openai
from metaphor_python import Metaphor
import API  # Replace with your own API keys or replace directly here
import requests
from bs4 import BeautifulSoup

# Replace with your API keys
openai.api_key = API.openAI_API
metaphor = Metaphor(API.metaphor_API)

# Function to suggest documentation URLs using Metaphor
def suggest_documentation_urls(document_name):
    query = f"Documentation for {document_name}"
    search_response = metaphor.search(query, use_autoprompt=True)
    return search_response.results[:5]

# Function for document summarization
def document_summarizer(document_text):
    #print(document_text)
    """
    completion = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": document_text},
        ],
    )
    return completion.choices[0].message.content
    """

def fetch_document_content(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            main_content = soup.find('div', {'class': 'main-content'}) 
            if main_content:
                document_text = main_content.get_text()
                return document_text
            else:
                return "Document content not found on the page."

        else:
            return "Failed to retrieve document content. Please check the URL."

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main program
if __name__ == "__main__":
    print("Welcome to the Documentation Summarizer!")

    document_name = input("Enter the name or topic of the documentation you want to summarize: ")
    documentation_urls = suggest_documentation_urls(document_name)

    print("\nTop documentation suggestions:")
    for idx, doc in enumerate(documentation_urls, start=1):
        doc_name = doc.title
        doc_url = doc.url
        print(f"{idx}. {doc_name}, {doc_url}")

    choice = input("\nSelect a documentation source (enter the number): ")
    selected_doc = documentation_urls[int(choice) - 1]

    doc_content = fetch_document_content(selected_doc.url)
    summary = document_summarizer(doc_content)

    print("\nSummary of the documentation:")
    print(summary)

    print("\nThank you for using the Documentation Summarizer!")
