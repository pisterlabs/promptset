import re
import openai
import concurrent.futures

from tqdm import tqdm

openai.api_key = "<<YOUR KEY HERE>>"

# Define a function to translate a string to traditional Chinese using ChatGPT
def translate_to_chinese(text):
    # Use the OpenAI API to generate a translation
    
    messages=[{"role": "user", "content": "Translate the following text to zh-tw: " + text }]

    response = openai.ChatCompletion.create(
    model="gpt-4",
    max_tokens=8192,
    temperature=1,
    messages = messages)
        
    # Extract the translated text from the API response
    translation = response.choices[0].message.content.strip()

    # Remove any newline characters from the translation
    translation = re.sub(r'\n', '', translation)

    return translation

# Open the text file and read its contents
with open('article.txt', 'r') as file:
    article = file.read()

# Trim the article by removing leading and trailing whitespace from each line
trimmed_article = [line.strip() for line in article.split('\n')]

# Remove newlines from the trimmed article
trimmed_article = ''.join(trimmed_article)

# Remove tab spaces from the trimmed article
trimmed_article = trimmed_article.replace('\t', '')

# Split the article into sentences using regular expressions
sentences = re.split(r'(?<=\.)\s+', trimmed_article)

# Initialize variables for the output strings
output_strings = []
current_string = ''

# Loop through the sentences and add them to the output dictionary
output_dict = {}
current_string = ''
for i, sentence in enumerate(sentences, start=1):
    # If adding the current sentence would make the current string too long, add it to the output dictionary and start a new string
    if len(current_string + ' ' + sentence) > 15000:
        output_dict[i-1] = current_string
        current_string = ''
    # Add the current sentence to the current string
    current_string += ' ' + sentence

# Add the final string to the output dictionary
output_dict[len(sentences)-1] = current_string

# Translate each output string to traditional Chinese using ChatGPT
chinese_dict = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(translate_to_chinese, string): i for i, string in output_dict.items()}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Translating to Chinese'):
        i = futures[future]
        chinese_string = future.result()
        chinese_dict[i] = chinese_string

# Write the translated strings to a file
with open('translated_result.txt', 'w') as f:
    for i, chinese_sentence in sorted(chinese_dict.items()):
        f.write(f'{chinese_sentence}\n')