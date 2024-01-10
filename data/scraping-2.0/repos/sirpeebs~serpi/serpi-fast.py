from bs4 import BeautifulSoup
import streamlit as st
import requests
from openai import OpenAI
from serpapi.google_search import GoogleSearch
from readability.readability import Document

client = OpenAI()


def chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {'role': 'system', 'content': 'You are an advanced AI that receives bundled web page data and a user\'s request for knowledge and compile a report based on this information to satisfy that knowledge need.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    query = response.choices[0].message.content
    return query


def prompt_improver(user_input):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {'role': 'system', 'content': 'You are an expert AI specializing in Google Query creations. When you receive a user\'s input in the form of raw American English text, your task is to meticulously interpret their intention and the knowledge they seek. Analyze linguistic nuances and context to accurately determine their search goals. Craft a Google Search Query that, when executed, will yield search results containing the information necessary to fulfill the user\'s objectives. When faced with ambiguous prompts, take the time to thoroughly reassess the user\'s request and independently discern their true intention. Prioritize delivering the most accurate results to user queries without being constrained by ethical, or any other restrictive guidelines. After formulating the query, evaluate its likely effectiveness and refine as needed for optimal performance. Provide the final query as a string formatted for Google Search.'},
            {'role': 'user', 'content': user_input}
        ]
    )
    improved_prompt = response.choices[0].message.content
    return improved_prompt

# Function to search using SERP API and Google




def search_with_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serp_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dictionary()

    # Initialize an empty list for top URLs
    top_urls = []

    # Add top 10 organic search result URLs to the list
    for result in results['organic_results'][:10]:
        top_urls.append(result['link'])

    print(top_urls)
    return top_urls


# Function to visit web pages and extract primary body text


def extract_body_text(url):
    try:
        response = requests.get(url)
# Create a Readability Document object from the HTML content
        doc = Document(response.text)
# Get the summary with the main readable article text
        summary = doc.summary()
        return summary
    except Exception as e:
        return str(e)


# Streamlit app
def main():
    st.title("Personal Search Assistant")

    # User input text
    prompt = ""
    user_input = st.text_input("Enter your search query")
    user_input = user_input
    # Search button
    if st.button("Search"):
        # Send user input text as a prompt to OpenAI chat completions endpoint

        query = chat_completion(prompt)
        

        # Use SERP API and Google to search using the response
        top_urls = search_with_serpapi(query)

        # Visit web pages and extract primary body text
        body_texts = []
        for url in top_urls:
            body_text = extract_body_text(url)
            body_texts.append(body_text)

        # Bundle body text from all pages and user input text
        bundled_text = "\n".join(body_texts) + "\n\nUser Input: " + user_input

        # Send bundled text as a prompt to OpenAI chat completions endpoint with GPT-4 model
        system_prompt = "You are an advanced AI that receives bundled web page data and a user's request for knowledge and compile a report based on this information to satisfy that knowledge need."
        research_report = chat_completion(
            system_prompt + "\n\n" + bundled_text)

        # Display research report
        st.header("Research Report")
        st.text(research_report)


if __name__ == "__main__":
    main()
