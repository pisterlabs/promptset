import openai
from .key import key
from bs4 import BeautifulSoup
import json
from datetime import datetime

openai.api_key = key
  
 # this is essentially a normal GPT function but with a timeout - i.e. if GPT takes too long to respond
 # the function will time out and return None to limit processing time 

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
        
        messages = [{"role": "system", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # NOTE: temp should be 0 since we're not trying to do anything creative here
            request_timeout=30  # request timeout of 15 seconds to prevent holdups
            )
        except Exception as e:
            print("GPT Error:")
            print(e)
            return None
        return response.choices[0].message["content"]

# extracts visible text from raw html and structures it with GPT
def prep_contents(element):

    soup = BeautifulSoup(str(element), 'lxml')
    # get all links in the HTML
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # all of this text processing is to remove anything that is not plain text in the html
    for script in soup(["script", "style"]):
            script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '-'.join(chunk for chunk in chunks if chunk)
    # structure contents 

    # check if the length of the extracted text is short to save GPT resources from random page elements scraped in error
    if len(text) < 20:
        print("Element too short. Removing...")
        return None 
        # this just creates the gpt prompt
    structured = structure_contents(text, links)
    return structured

# turn plain text into structured JSON via GPT
def structure_contents(contents, links):
    event_tags = [
    "Music", "Happy-hour", "Food", "Networking", "Art", "Workshop", "Sports", 
    "Charity", "Education", "Festival", "Outdoor", "Performance", "Lecture",
    "Seminar", "Conference", "Dance", "Film", "Theater", "Comedy", "Literature",
    "Family-friendly", "Cultural", "Fashion", "Craft", "Exhibition", "Fitness",
    "Yoga", "Meditation", "Technology", "Fundraiser", "Auction", "Launch", 
    "Celebration", "Spiritual", "Culinary", "Wine-tasting", "Beer-tasting", 
    "Pop-up", "DIY", "Virtual", "Adventure", "Travel", "Nightlife", "Rave", "Retreat",
    "Reunion", "Gaming", "Role-playing", "Cosplay", "Market", "Trade-show"
    ]

    now = datetime.now()

    short_prompt = f"""
Given the text and links below, provide a response following the provided JSON structure:

Text:
{contents}

Links:
{links}

JSON Format:
{{
  "is_event": "boolean",  # Set to 'True' ONLY if the text refers to ONE SPECIFIC event. Set to 'False' for general mentions or non-informative content.
  "title": "string",
  "description": "string",  # Brief summary.
  "start_date": "string",   # Format: YYYY-MM-DD or YYYY-MM-DD HH:MM. Default to {now.year}-{now.month} if partial.
  "end_date": "string",
  "location": "string",
  "price": "float",         # Use -1 if not mentioned, 0 if free.
  "sold_out": "boolean",
  "link": "string",         # Provide a relevant link from the 'links'.
  "img_link": "string",     # Provide a search string which is less specific than the title to locate an image relevant to the event, rather than a direct link. This cannot be null.
  "tags": ["string", ...],  # Use ONLY 1-3 appropriate tags from the list: {event_tags}
  "confidence": "integer"   # Confidence level (0-10) regarding the accuracy of event details.
}}

If any attribute is unavailable, use "attribute_name": null. Ensure every attribute is included. Stick to the date format: YYYY-MM-DD or YYYY-MM-DD HH:MM.
"""
    # tags and GPT prompt
   
    return short_prompt

# helper function to safely parse JSON output from GPT
# also converts some text to ensure the JSON is compliant
def parse_json(input):
    try:
        # ensure common errors are fixed. Ex: false -> False
        valid_json_string = (input.replace("Null", "null")
                    .replace("False", "false")
                    .replace("True", "true"))
        event = json.loads(valid_json_string)
        return event
    except Exception as e: # return None if error
        print("Error converting GPT ouput to JSON")
        print(e)
        return None