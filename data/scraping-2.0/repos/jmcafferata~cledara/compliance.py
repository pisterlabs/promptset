import openai
import requests
from xml.etree import ElementTree
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from ast import literal_eval


def can_fetch_robots(robots_url, user_agent, url_to_check):
    """Checks the robots.txt for permissions."""
    response = requests.get(robots_url)
    lines = response.text.split("\n")
    
    allowed, disallowed = [], []
    active_user_agent = None

    for line in lines:
        line = line.strip().lower()
        
        if line.startswith('user-agent:'):
            agent_name = line.split(':', 1)[1].strip()
            if agent_name == '*' or agent_name == user_agent.lower():
                active_user_agent = agent_name
        elif active_user_agent:
            if line.startswith('allow:'):
                allowed.append(line.split(':', 1)[1].strip())
            elif line.startswith('disallow:'):
                disallowed.append(line.split(':', 1)[1].strip())
                
    for path in disallowed:
        if url_to_check.endswith(path):
            return False
    for path in allowed:
        if url_to_check.endswith(path):
            return True
            
    return True

def fetch_links_from_homepage(url):
    print(f"Fetching links from {url}")
    """Fetch all links from the given homepage."""
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def get_relevant_links_via_openai(homepage_url,links):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """Example prompt:\n
            The website is https://cledara.com. From this list of links, pick the URL where I may find the terms and conditions and privacy policy: ['/terms', '/privacy', 'https://cledara.com/blog','https://cledara.com/cookie-policy']\n
             Example answer: \n
             ['https://cledara.com/terms', 'https://cledara.com/privacy', 'https://cledara.com/cookie-policy']"""},

            {"role": "user", "content": f"The website is {homepage_url}. From this list of links, pick the URL where I may find the terms and conditions and privacy policy: {links}"}
        ]
    )
    
    print(response.choices[0].message['content'])
    return literal_eval(response.choices[0].message['content'])

# def get_information_from_web(urls,data_request):
#     for url in urls:
#         #get text from url
#         response = requests.get(url)
#         #get text from page using utf-8 
#         response.encoding = 'utf-8'
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         text = soup.get_text()
#         # split the text in chunks of 5000 characters
#         chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
#         for i, chunk in enumerate(chunks):
#             print(f"Chunk {i+1} of {len(chunks)}")
#             gdpr_response = openai.ChatCompletion.create(
#                 model = 'gpt-3.5-turbo',
#                 messages = [
#                     {"role": "system", "content": f"""
#                      I need this data:\n{data_request}\nRead the text the user gives you and respond with the previous list with the information added. If the field is already full leave it or update it."""},
#                      {"role": "user", "content": "Reading "+ url+ "\n"+chunk}]
#             )
#             data_request = gdpr_response.choices[0].message['content']
#             print(data_request)

#     final_response = openai.ChatCompletion.create(
#         model = 'gpt-3.5-turbo',
#         messages = [
#             {"role": "system", "content": "Convert this table into a nice html table, starting with <table> and ending with </table>"},
#                 {"role": "user", "content": gdpr_response.choices[0].message['content']}]
#     )
#     print(final_response.choices[0].message['content'])
#     return final_response.choices[0].message['content']

# homepage_url = "https://tropicapp.io"
# all_links = fetch_links_from_homepage(homepage_url)
# relevant_links = get_relevant_links_via_openai(homepage_url,all_links)
# print(relevant_links)
# data_request = """

# """
# data_response = get_information_from_web(relevant_links, data_request)

# print("Here's the data I got:")
# print(data_response)
