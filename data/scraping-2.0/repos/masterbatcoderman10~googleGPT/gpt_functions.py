import os
import json
import openai
import pprint
from dotenv import load_dotenv
from google_search import *
import tiktoken

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')


def get_query(input_query):

    query_message = {
    "role": 'system',
    "content": """
        You are a subsystem within a larger system designed to search the web.
        Your role is to craft the most relevant and comprehensive query based on the input provided.
        Return the query STRICTLY as a dictionary with the key: 'optimized_search_query'.
        Here's the format: {"optimized_search_query": "your refined query here"}.
    """
    }


    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.9,
        messages=[query_message, {"role": 'user', "content": input_query}],
    )

    content = response['choices'][0]['message']['content']
    # turn content string into json
    content = json.loads(content)

    return content


def decide_leads(links, query):

    """
    Determines the most suitable links for a given query using the OpenAI GPT-3 model.

    This function sends a system message and a user message to the GPT-3 model. 
    The system message instructs the model to evaluate the provided links and determine which ones 
    are the most suitable for finding information related to the given query. 
    The user message contains the query and the links. 
    The model's response is then parsed and returned as a dictionary.

    Args:
        links (list): The list of links to be evaluated.
        query (str): The query to find information for.

    Returns:
        dict: A dictionary containing the most suitable links. The key is 'link' and the value is the selected link.
    """

    sys_message = {
        'role': 'system',
        'content': """
        Your task is to evaluate the provided links and determine which ones 
        are the most suitable for finding information related to the given query. 
        Return the most most suitable links as a JSON object with the key "link", 
        Just a string JSON no backticks.
        """
    }

    user_message = {
        'role': 'user',
        'content': f"The query is {query}. The links are ```{links}```."
    }

    messages = [sys_message, user_message]

    input_tokens = num_tokens_from_messages(messages)
    print(input_tokens)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.0,
        messages=messages,
    )

    content = response['choices'][0]['message']['content']
    content = json.loads(content)
    return content
# pprint.pprint(get_query('what is apple latest product that is released?'))


def summarize(query, webpage):
    """
    Summarizes the content of a webpage based on a user's query using the OpenAI GPT-3 model.

    This function sends a system message and a user message to the GPT-3 model. 
    The system message instructs the model to summarize the content of the provided webpage based on the user's query. 
    The user message contains the query and the webpage content. 
    The model's response is then parsed and returned as a dictionary.

    Args:
        query (str): The user's query.
        webpage (str): The content of the webpage to be summarized.

    Returns:
        dict: A dictionary containing the summary. The key is 'summary' and the value is the summary of the webpage content.
    """
    sys_message = {
        'role': 'system',
        'content': """
            Your task is to summarize the content of the provided webpage based on the user's query. 
            The summary should be clear, well-explained, and original - do not copy text directly from the webpage. 
            The summary must be returned as a JSON object with the key "summary". 
            Here's the format: {"summary": "your summary here"}.
            If the information needed to answer the query isn't available on the provided webpage, 
            return a JSON object with the value "N/A" for the summary key, like so: {"summary": "N/A"}.
            """
    }

    enc_webpage = encoding.encode(webpage)
    #truncate the encoding to 2048 tokens
    enc_webpage = enc_webpage[:2048]
    webpage = encoding.decode(enc_webpage)

    user_message = {
        'role': 'user',
        'content': f"The query is {query}. The webpage is ```{webpage}```."
    }

    messages = [sys_message, user_message]

    input_tokens = num_tokens_from_messages(messages)
    print(input_tokens)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.0,
        messages=messages,
    )

    content = response['choices'][0]['message']['content']
    content = json.loads(content)
    return content


# test_links = {"Apple Event: what's happened in 2023 and what could be next - TechRadar": 'https://www.techradar.com/news/new-apple-event',
#  'Apple shows off MacBooks with M3 chip at Scary Fast event - Yahoo Finance': 'https://finance.yahoo.com/video/apple-shows-off-macbooks-m3-140203045.html',
#  'Apple unveils new MacBook Pro featuring M3 chips': 'https://www.apple.com/newsroom/2023/10/apple-unveils-new-macbook-pro-featuring-m3-chips/',
#  "Apple's 'Scary Fast' Mac event: all the news from Apple's online keynote - The Verge": 'https://www.theverge.com/2023/10/30/23933672/apple-event-mac-october-news-updates-products-scary-fast',
#  'Apple: News, Reviews, Guides, and More - PCMag': 'https://www.pcmag.com/series/apple',
#  'Every new Apple product coming in 2024 - Macworld': 'https://www.macworld.com/article/671090/new-apple-products.html',
#  'The new MacBook Pro | Apple - YouTube': 'https://youtube.com/watch?v=0pg_Y41waaE',
#  'Upcoming Apple Products Guide: Everything We Expect to See in 2023 and Beyond': 'https://www.macrumors.com/guide/upcoming-apple-products/'}

# pprint.pprint(decide_leads(test_links, 'Apple latest product released'))

# pprint.pprint(summarize('what is apple latest product that is released?', """The iPhone 15 Pro and iPhone 15 Pro Max were this year's headlining act, with new and improved form factors, updated processors, and the latest advancements in the iPhone camera.

# The biggest change, at least on the surface, is the slimmer and lighter design, which has new contoured edges, Grade 5 Titanium, and the thinnest borders ever on an iPhone.

# Also: iPhone 15 Pro hands-on: I found a lot of reasons to upgrade and one to wait until next year

# The iPhone 15 Pro features the fastest mobile CPU -- A17 Pro. The new CPU can run up to 10% faster than the A16 Bionic chip in last year's iPhone 14 Pro. Compared to Apple's previous iterations, the neural engine is up to two times faster, and the pro-class GPU is up to 20 times faster, optimizing overall smartphone performance.

# As for the cameras, the Pro variants feature a 48MP main camera, which allows users to switch between the 24mm, 28mm, and 35mm focal lengths and supports 48MP HEIF images with up to four times the resolution. The Pro Max also features a 5X telephoto lens, the longest optical zoom ever on iPhone.

# Like on the iPhone 15 and iPhone 15 Plus, Portrait Mode now runs in the background, so the iPhone can measure the depth information of a photo's subject and later give users the ability to adjust the bokeh effect. The quality of photos in low light will also improve, with a night mode that provides sharper detail and more vivid colors.

# Also: 4 key features the iPhone 15 Pro is still missing

# Besides the cameras, another major selling point is the Action Button, which replaces the traditional alert slider. The Action Button will function as a ring and silent button by default but can be personalized to be a mappable quick key for turning on the camera app, flashlight, Siri, Shortcuts, and more.

# In terms of safety, the phones feature Crash Detection, Emergency SOS via satellite, and a new Roadside Assistant that allows customers to comment to AAA when experiencing car troubles.

# Also: Apple iPhone 15 Pro mutes side switch for multifunctional Action button

# With the EU imposing a law for electronics makers to adopt USB-C by 2024, the new iPhones finally support charging and data transferring via USB-C, becoming Apple's latest major product to make the switch to the more universal power format.

# The phones are available for preorder now in black titanium, white titanium, blue titanium, and natural titanium finishes. The iPhone 15 Pro starts at $999, and the iPhone 15 Pro Max starts at $1,199."""))

def craft_payload(query):
    sys_message = {
        'role': 'system',
        'content': """
            You are a sophisticated digital surfer with a knack for crafting precise Google search queries.
            Based on the input query, your task is to formulate a refined search query by selecting the most 
            relevant advanced search parameters. Your goal is to return a payload dictionary optimized for 
            an accurate and efficient search experience. Analyze the input query, deduce the intent, and 
            choose the search parameters wisely to hone in on the most pertinent information.

            The payload should be returned as a dictionary with the following keys:
            - 'q': The refined query.
            - 'num': The number of results to retrieve (default to 10).
            - 'tbs': Time-based search parameter (e.g., 'qdr:w' for past week), if relevant.
            - 'as_filetype': Filetype filter (e.g., 'pdf'), if relevant.
            - 'as_sitesearch': Specific domain to restrict the search to (e.g., 'example.com'), if relevant.
            - 'cd_min' : the start data
            - 'cd_max' : the end date

            Here's the expected format:
            {
                "q": 'refined query here',
                "num": 10,
                "tbs": 'time-based parameter here',
                "cd_min": 'start date here',
                "cd_max": 'end date here',
                "as_filetype": 'filetype here',
                "as_sitesearch": 'domain here'
            }

            Omit any keys that are not relevant to the refined search query.
        """
    }

    user_message = {
        'role': 'user',
        'content': query
    }

    messages = [sys_message, user_message]

    input_tokens = num_tokens_from_messages(messages)
    print(input_tokens)

    # Set up the input for the OpenAI call

    response = openai.ChatCompletion.create(
        model='gpt-4',
        temperature=0.0,
        messages=messages,
    )
    
    # Extract the payload from the response
    payload = response['choices'][0]['message']['content']
    # Turn the payload string into a JSON object
    payload = json.loads(payload)

    return payload


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



