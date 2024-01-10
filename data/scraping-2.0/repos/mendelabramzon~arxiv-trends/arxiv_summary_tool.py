import requests
import json
import openai
import argparse

# Initialize OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'

ARXIV_API_URL = "http://export.arxiv.org/api/query?"

def fetch_recent_articles_from_arxiv(keyword, max_results=30):
    # Construct the query for articles related to the keyword
    query = f"search_query=all:{keyword}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={max_results}"
    response = requests.get(ARXIV_API_URL + query)
    
    # Parse the XML response to extract abstracts
    entries = response.text.split('<entry>')
    abstracts = [entry.split('<summary>')[1].split('</summary>')[0].strip() for entry in entries[1:]]
    
    return abstracts

def summarize_with_openai(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k-0613",
      messages=[
            #{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me the summary for the following collection of abstracts: {text}"}
        ],
      max_tokens=500  # Adjust as needed
    )
    return response.choices[0]['message']['content']

def main():
    parser = argparse.ArgumentParser(description="Fetch recent articles from arXiv based on a keyword and generate a summary.")
    parser.add_argument('keyword', type=str, help="Keyword to search for in arXiv.")
    
    args = parser.parse_args()

    abstracts = fetch_recent_articles_from_arxiv(args.keyword)
    combined_abstracts = " ".join(abstracts)
    summary = summarize_with_openai(combined_abstracts)
    
    print(f"Summary for articles related to '{args.keyword}':\n{summary}")

if __name__ == "__main__":
    main()
