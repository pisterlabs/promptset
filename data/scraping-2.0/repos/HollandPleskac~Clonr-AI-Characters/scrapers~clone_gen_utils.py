import io
import os
import guidance
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from clonr.data.parsers import FullURLParser, FandomParser
from clonr.data.parsers import WikiQuotesParser
from clonr.data.parsers import WikipediaParser
import requests
from .utils import find_char_url, find_links_with_same_base_url
from .characterai_utils import parse_results

def generate_example_quotes(char_name, char_title, char_greeting, char_category):
    load_dotenv()

    example_quotes = """
    {{#system~}}
    You are a helpful AI assistant that can generate character profiles.
    {{~/system}}
    {{#user~}}
    Please answer the following questions to generate a character profile based on the given information.

    Here is the name of the character: {{char_name}}

    Here is the title of the character: {{char_title}}

    Here is an example greeting of the character: {{char_greeting}}

    Here is the category of the character: {{char_category}}

    Could you produce five example quotes that the character would say?

    {{~/user}}
    {{#assistant~}}
    {{gen 'result' temperature=0.1 max_tokens=1000}}
    {{~/assistant}}
    """

    gpt_turbo = guidance.llms.OpenAI('gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    char_name = 'Raiden Shogun and Ei'
    char_title = 'From Genshin Impact'
    char_greeting = 'Shogun: No salutations needed. My exalted status shall not be disclosed as we travel among the common folk. I acknowledge that you are a person of superior ability. Henceforth, you will be my guard. Worry not. Should any danger arise, I shall dispose of it.'
    char_category = 'Anime Game Characters'
    example_quotes = guidance(example_quotes, llm=gpt_turbo)

    result = example_quotes(char_name=char_name, char_title=char_title, char_greeting=char_greeting, char_category=char_category)
    print(result)
    return result


def generate_character_profile(char_data):
    char_title = char_data['title']
    char_name = char_data['participant__name']
    char_greeting = char_data['greeting']
    char_category = char_data['character_category']
    avatar_file_name = char_data['avatar_file_name']

    print("This is char_name: ", char_name)
    if 'from' in char_title.lower():
        char_wiki = char_title.lower().split("from ")[1]
        char_wiki = char_wiki.replace(" ", "-")
    else:
        return 
    print("This is char_wiki: ", char_wiki)
    fandom_parser = FandomParser()
    fandom_content = None
    try:
        fandom_result = fandom_parser.extract(char_name, char_wiki)
        fandom_content = fandom_result.content
    except Exception as e:
        print("Cannot get Fandom result: ", e)

    wikiquotes_parser = WikiQuotesParser()
    wikiquotes_content = None
    try:
        wikiquotes_result = wikiquotes_parser.extract(char_name)
        wikiquotes_content = wikiquotes_result.content
    except Exception as e:
        print("Cannot get WikiQuotes result: ", e)

    if wikiquotes_content is None:
        print("Generating synthetic quotes..")
        #generate_example_quotes()
    
    img_prefix = 'https://characterai.io/i/80/static/avatars/'
    img = None
    
    # try:
    #     img = download_webp_image(img_prefix + avatar_file_name)
    # except Exception as e:
    #     print("Cannot download image: ", e)

    return {
        fandom_content: fandom_content,
        wikiquotes_content: wikiquotes_content,
        img: img,
    }

def extract_quotes_from_url(parser, url):
    fandom_result = parser.extract(url)
    if fandom_result:
        return fandom_result.content
    else:
        return None

# Get img url from fandom
def get_character_image(base_url):
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img_anchor = soup.find('a', {'class': 'image image-thumbnail', 'title': 'Card'})
            
            if img_anchor and 'href' in img_anchor.attrs:
                image_url = img_anchor['href']
                return image_url
    except requests.exceptions.RequestException as e:
        print("Error fetching the webpage:", e)
    
    return None

# Getting example dialogues
def get_all_example_dialogues(char_name, char_wiki, parser):
    print(f"In get_all_example_dialogues, for char_name: {char_name}, char_wiki: {char_wiki}")
    char_url = find_char_url(char_name, char_wiki)
    results = []
    print("this is char_url: ", char_url)
    if char_url:
        found_links = find_links_with_same_base_url(char_url)
        print("Links on the webpage that contain the original webpage URL:")
        print(found_links)
        for found_link in found_links:
            print("Processing link: ", found_link)
            result = extract_quotes_from_url(parser, found_link)
            if result:
                results.append(result)
    
    results = "\n\n".join(results)

    if len(results) < 100:
        print("No results found for this character.")
        return None

    file_path = f'scrapers/fandom/{char_name}_{char_wiki}.txt'

    if not os.path.exists(file_path):
        with open(file_path, 'x') as f:
            f.write(results)
    else:
        print(f"File '{file_path}' already exists. Won't overwrite.")
    return results

# parser = FullURLParser()

# results = parse_results()

# for result in results:
#     char_name = result['participant__name']
#     char_title = result['title']
#     char_wiki = ''
#     if 'from' in char_title.lower():
#         char_wiki = char_title.lower().split("from ")[1]
#         char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')
#     if 'of' in char_title.lower():
#         char_wiki = char_title.lower().split("of")[1].strip("of ")
#         char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')
    
#     if char_wiki == '':
#         continue 

#     print(f"Processing char_name = {char_name}, char_wiki = {char_wiki}")
#     total_results = get_all_example_dialogues(char_name, char_wiki, parser)
