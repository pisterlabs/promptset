import openai
import os
from metaphor_python import Metaphor

OPENAI_API_KEY_FILE_PATH = "/Users/anahitasrinivasan/Desktop/APIKey/api-key.txt"
OPENAI_ORGANIZATION_FILE_PATH = "/Users/anahitasrinivasan/Desktop/APIKey/organization.txt"
METAPHOR_API_KEY_FILE_PATH = "/Users/anahitasrinivasan/Desktop/APIKey/metaphor-api-key.txt"


def read_api_key(file_path):
    '''
    Returns a stripped, string version of an API key given the file path.
    '''
    result = ""
    with open(file_path) as f:
        result = f.readline().strip()
    return result


# Setting up both ChatGPT and Metaphor with API keys and organization tokens.
metaphor = Metaphor(read_api_key(METAPHOR_API_KEY_FILE_PATH))
openai.api_key = read_api_key(OPENAI_API_KEY_FILE_PATH)
openai.organization = read_api_key(OPENAI_ORGANIZATION_FILE_PATH)

# type_of_food = "Indian"
# city = "Fremont"

# Ask users for both the type of food they're looking for and the city they're
# searching in.
type_of_food = input("Enter what type of food you're looking for: ").strip()
city = input("Enter what city you're searching in: ").strip()

USER_QUESTION = "Where can I find the best " + \
    type_of_food + " restaurants in " + city + "?"
SYSTEM_MESSAGE_1 = "You are a helpful assistant that turns questions into search queries. Only generate one search query."

# Asking ChatGPT to reformat the user question into a search query.
prompt = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE_1},
        {"role": "user", "content": USER_QUESTION},
    ],
)

# Pass the query into Metaphor
query = prompt.choices[0].message.content
search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2016-01-01"
)

# Grab the contents of the query and create a set to prevent duplicate results from being
# returned.
all_contents = search_response.get_contents().contents
seen = set()

SYSTEM_MESSAGE_2 = "You are a helpful assistant that pulls out the restaurant names given the webpage title and URL. Return only the restaurant names."
SYSTEM_MESSAGE_3 = "You are a helpful assistant who summarizes a restaurant description. Return only the summary of up to 200 characters."
print("\nHere are some of the best " +
      type_of_food + " restaurants in " + city + ".\n")


for item in all_contents:
    # Format content for ChatGPT that provides both the title and URL for a particular
    # restuarant's home page
    user_content = "The title is " + item.title + ". The URL is " + item.url + "."

    # pass the content into ChatGPT
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE_2},
            {"role": "user", "content": user_content},
        ],
    )

    # Get the restaurant name from ChatGPT
    name = completion.choices[0].message.content

    # Pass an extract from the restaurant website into ChatGPT and ask it to construct
    # a brief summary of the restaurant
    description = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE_3},
            {"role": "user", "content": item.extract},
        ],
    )

    # This attempts to weed out search results that aren't restaurant names, such as
    # "20 best restaurants in Boston" or other similar compilation articles
    if len(name) < 30:
        # This attempts to weed out duplicate search restuls.
        if name not in seen:
            seen.add(name)
            # Print out the restaurant name, URL, and a brief summary.
            print(name)
            print(item.url + "\n")
            print(description.choices[0].message.content + "\n")
