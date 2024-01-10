import cloudscraper
from bs4 import BeautifulSoup
import re
import openai
from tavily import TavilyClient

# Initialize the Tavily API Client and OpenAI API key
tavily = TavilyClient(api_key="tvly-xxxx")
openai.api_key = 'sk-xxxx'

# Function to get URLs and contents using Tavily API with specified settings
def get_urls_and_contents(query):
    response = tavily.search(query=query, search_depth="advanced", max_results=5, include_answer=False, include_images=False, include_raw_content=False, include_domains=["soccerstats.com"], exclude_domains=[])
    print("Tavily API Response:", response)
    if 'results' in response:
        return [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    else:
        print("KeyError: The response does not contain 'results'.")
        return []

# Function to get main content from a URL using cloudscraper and BeautifulSoup
def get_main_content(url):
    scraper = cloudscraper.create_scraper(delay=10, browser='chrome')
    try:
        response = scraper.get(url).text
        soup = BeautifulSoup(response, 'html.parser')
        extracted_text = re.sub(r'\s+', ' ', soup.get_text()).strip()
        print(f"Extracted content from {url}:\n{extracted_text}\n")
        return extracted_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "BLOCKED" if "Cloudflare" in str(e) else None

# Function to clean content with gpt-4-1106-preview
def get_cleaned_content_from_gpt(content, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a crypto analysis editor."},
                {"role": "user", "content": f"Detailed analysis of XEN coin:\n\n{content}"}
            ],
            temperature=0,
            max_tokens=1500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Error processing content with gpt-3.5-turbo-16k: {e}")
        return content

# Execute the scraping and cleaning process
query = "Detailed analysis of XEN coin"
url_contents = get_urls_and_contents(query)
combined_cleaned_content = ""  # Initialize variable to store combined content

for item in url_contents:
    url = item["url"]
    full_content = get_main_content(url)

    cleaned_content = get_cleaned_content_from_gpt(full_content, query)
    combined_cleaned_content += cleaned_content + "\n\n"  # Append the cleaned content

# Print the final combined content
print("Final Combined Cleaned Content:\n", combined_cleaned_content)
