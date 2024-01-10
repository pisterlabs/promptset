import openai
from api_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    """
    On temperature:
    0: reliable, but conservative
    1: creative, but unpredictable
    
    When choosing the next word in the sequence, the model will sample
    from the distribution of words.

    A temperature of 0 means it will always choose the most likely word.
    A temperature of 1 means it will randomly sample from the full distribution.
    """

    return response.choices[0].message["content"]

url = "https://en.wikipedia.org/wiki/Bicycle"

# Using gpt-3.5-turbo to summarize a wikipedia page from just URL
# Note: The result is a bit rubbish - wrong headings (some made up, some missing)
prompt0 = f"""
Here is a some text: '{url}'

If this text is not a valid wikipedia URL, give me the response:

'This is not a wikipedia URL. Please try again.'

If it is a valid wikipedia URL, summarize the web page in plain text by doing
the following:

1. Get the html from the wikipedia page with the URL
2. Parse it to get the title of the page, the section headings, and the text of each section.
3. Summarize each section into a single sentence.

Now print a plain text summary of the wikipedia page in the following format:

<Title>
<Section Heading 1>: <Summary of Section 1>
<Section Heading 2>: <Summary of Section 2>

etc. for each section.

If there is a 'External links' section, 'References' section, 'Further reading',
or 'See also' section, do not include it in the summary.

If unable to get the html from the wikipedia page, return the response:

'Unable to get the html from the wikipedia page. Please try again.'
"""
# print(get_completion(prompt=prompt0))


# A mixture of web scraping and gpt-3.5-turbo to summarize a wikipedia page.
import requests
from bs4 import BeautifulSoup

if 'wikipedia.org' not in url:
    print('This is not a wikipedia URL. Please try again.')
else:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1', {'id': 'firstHeading'}).text
    sections = soup.find_all('span', {'class': 'mw-headline'})
    summary = ''
    # Record sections so as not to go over API rate limit! Free plan gives 3
    # requests per minute.
    section_num = 0
    for section in sections:
        if section.text and section.text not in ['External links', 'References', 'Further reading', 'See also']:
            section_text = section.find_next('p')
            if section_text:
                section_text = section_text.text
                prompt = (
                    f"Summarize the below text into a single sentance:\n\n"
                    f"```\n{section_text}\n```"
                )
                # if section_num < 2:
                #     section_summary = get_completion(prompt=prompt) # Disabling to save API calls
                #     section_num += 1
                # else:
                #     section_summary = "Oops, hit the rate limit!"
                section_summary = section_text
                summary += f'{section.text}: {section_summary}\n'
    print(f'{title}\n{summary}')
