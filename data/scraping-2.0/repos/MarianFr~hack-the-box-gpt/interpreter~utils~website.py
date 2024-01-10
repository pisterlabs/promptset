import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

fetched_urls = set()
all_reports = []

def save_content(content, filepath, api_key, max_length=16000000):
    """
    Save content to a specified file path, dividing it into parts if it exceeds the maximum length.

    Parameters:
        content (str): Textual content to save.
        filepath (str): The target file path where content should be saved.
        max_length (int, optional): Upper limit before content is split into parts. Default is 16,000,000.

    Notes:
        If content exceeds max_length, it will be divided into parts and saved in separate files.
        Reports from analysis (if any) are added to the global `all_reports` list and printed.
    """

    if len(content) > max_length:
        # Split content into parts
        num_parts = len(content) // max_length + (1 if len(content) % max_length > 0 else 0)
        for i in range(num_parts):
            part_filepath = f"{filepath.rsplit('.', 1)[0]}_part_{i+1}.{''.join(filepath.rsplit('.', 1)[1:])}"
            with open(part_filepath, 'w', encoding='utf-8') as file:
                file.write(content[i * max_length: (i + 1) * max_length])
            print(f"Content saved at {part_filepath}")
    else:
        # Save normally if content is short enough
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Content saved at {filepath}")

    # Example API key retrieval from an environment variable
    openai_api_key = api_key
    
    report_message = analyze_content_with_chatgpt(content[:3000], openai_api_key)  # Ensure within token limit
    
    # Log or otherwise use the generated report
    if report_message:
        report = {
            "filepath": filepath,
            "message": report_message,
        }
        all_reports.append(report)
        print(f"ChatGPT Report: {report['message']}")


def save_asset(url, folder, api_key):
    """
    Download and save a web asset (like CSS, JS, HTML files or images) locally.

    Parameters:
        url (str): URL of the asset to be saved.
        folder (str): Local folder where the asset should be saved.

    Notes:
        - Textual content like HTML, CSS, JS might be split into parts if they are too long (see `save_content`).
        - Binary content (e.g., images) will be saved normally without splitting.
    """

    print(f"Saving asset {url} to {folder}")
    response = requests.get(url)
    filename = url.split('/')[-1]
    filepath = os.path.join(folder, filename)
    
    if filepath.endswith(".html") or filepath.endswith(".css") or filepath.endswith(".js"):
        # Text content (HTML, CSS, JS), check and possibly split
        content = response.text
        save_content(content, filepath, api_key)
    else:
        # Binary content (images, etc.), save normally
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"Asset saved at {filepath}")


# Function to fetch website recursively
def fetch_website(base_url, path, api_key, hostname=None):
    """
    Recursively fetch website content starting from a base URL.

    Parameters:
        base_url (str): The base URL to start fetching from.
        path (str): Path to the resource to fetch from base_url.
        hostname (str, optional): Hostname used in HTTP headers when making requests.

    Notes:
        - Website content is fetched, parsed, and stored locally.
        - Assets like CSS, JS, and images are extracted and stored.
        - Internal links are identified and recursively fetched.
        - Once all fetching is completed, a summary report is saved.
    """
        
    # Form the complete URL
    url = urljoin(base_url, path)
    if url in fetched_urls:
        print(f"URL {url} has already been fetched, skipping.")
        return
    print(f"Fetching {url}")
    fetched_urls.add(url)
    
    # Prepare headers
    headers = {}
    if hostname:
        headers["Host"] = hostname
    
    # Fetch the HTML content
    response = requests.get(url, headers=headers)
    html_content = response.text
    
    # Parse HTML Content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Save HTML content
    folder = os.path.join('website', path.lstrip('/'))
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, 'index.html')
    save_content(html_content, filepath, api_key)  # Using the new function
    
    # Extract and save assets
    scripts = [script.get('src') for script in soup.find_all('script') if script.get('src')]
    styles = [link.get('href') for link in soup.find_all('link') if link.get('href')]
    # images = [img.get('src') for img in soup.find_all('img') if img.get('src')]  # Commented out
    
    # Instead of appending images to the asset list, just use scripts and styles
    for asset_url in scripts + styles:  # Removed + images
        save_asset(urljoin(base_url, asset_url), folder, api_key)
    
    # Extract internal links and fetch them recursively
    internal_links = [a.get('href') for a in soup.find_all('a', href=True) if a.get('href').startswith('/')]
    for link in internal_links:
        fetch_website(base_url, link)

    summarized_reports = "\n".join([f"File: {r['filepath']},Message: {r['message']}..." for r in all_reports])  # Truncating messages for brevity
    report_filepath = os.path.join('website', 'Ranking_Report_of_Fetcheed_Files_LOOK_AT_FIRST.txt')
    with open(report_filepath, 'w', encoding='utf-8') as report_file:
        report_file.write("Ranking Report of Fetcheed Files. LOOK AT FIRST:\n")
        report_file.write(summarized_reports)
        print(f"Ranking report saved to: {report_filepath}")



import openai

def analyze_content_with_chatgpt(content, openai_api_key):
    """
    Analyze content with ChatGPT and generate a report.

    Parameters:
        content (str): The content to analyze.
        openai_api_key (str): Your OpenAI API key.

    Returns:
        dict: A report generated by GPT.
    """
    # Ensure API key is provided
    if not openai_api_key:
        raise ValueError("No OpenAI API key provided")
    
    # Constructing the chat-based prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes HTML and other web files to find potentially entery points, that could be exploited, to gain user credentials or find other entry points of the machine its running on. Rate the content on a scale from 1-20 based on its potential relevance to a hack the box CTF challenge and provide a brief summary."},
        {"role": "user", "content": content[:3000]}  # Limiting to 3000 chars for demonstration; adjust as needed
    ]
    
    # Send API request to ChatGPT
    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=messages,
          temperature=0 # Adjust as needed
        )
        
        # Process the response
        message = response['choices'][0]['message']['content'].strip()
        return message
    
    except Exception as e:
        print(f"Error calling ChatGPT: {str(e)}")
        return None

# # # Start fetching the website from the base URL
# ip_address = "10.10.11.221"
# # hostname = "2million.htb"
# url = f"http://{ip_address}"

# # print("Starting to fetch the website...")
# fetch_website(url, '/')
# # print("Fetching completed.")
