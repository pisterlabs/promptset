import configparser
import openai
from modules.errors import robust_api_call
from tqdm import tqdm

# Load the configuration file
config = configparser.ConfigParser()
config.read('modules/suite_config.ini')

# Access variables
use_tqdm = config.getboolean('General', 'UseTqdm')
model = config['Models']['CategorizeHeadlines']
categories = config['Headlines']['Categories']
openai_api_key = config['OPENAI']['OPENAI_API_KEY']

openai.api_key = openai_api_key

def categorize_headlines(headlines):
    categorized_headlines = []

    iterator = tqdm(headlines, desc="Categorizing headlines") if use_tqdm else headlines

    for headline in iterator:
        # Create the prompt as a system message to instruct the model to return only one category
        system_message = f"Classify the following headline into a single category: {categories}. Provide only one category that best fits the headline."

        # User message is the headline to classify
        user_message = headline[0]

        # Prepare the conversation with system and user messages
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        response = robust_api_call(lambda: openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            request_timeout = 30
        ), retries=3, base_delay=2)


        if response is not None:
            # Parse the category from the response
            category = response['choices'][0]['message']['content'].strip()
            print(f"Model response: {category}")
            # Ensure only one category is appended by splitting and taking the first one if multiple are provided
            categorized_headlines.append((headline[0], category.split(',')[0].strip(), headline[1], headline[2], headline[3]))
        else:
            print("Failed to get category for a headline after all retries. Skipping...")
            categorized_headlines.append((headline[0], None, headline[1], headline[2], headline[3]))

    return categorized_headlines


