import openai
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
import os
import dotenv
import json

# Load the .env file
dotenv.load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Make sure you've downloaded the required nltk packages
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Get the message from the user
user_message = input("Please enter your message: ")

# Tokenize the message, identify past tense verbs and pronouns
tokens = nltk.word_tokenize(user_message)
tagged = nltk.pos_tag(tokens)

# Change past tense verbs to present tense and replace pronouns
new_tokens = []
for word, tag in tagged:
    if tag == 'VBD':  # This is the tag for past tense verbs
        present_tense = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        new_tokens.append(present_tense)
    elif tag in ['PRP', 'PRP$', 'WP', 'WP$']:  # These are the tags for pronouns
        new_tokens.append('individual')
    else:
        new_tokens.append(word)

# Join the tokens back into a message
new_message = ' '.join(new_tokens)

# Use RAKE to extract keywords
r = Rake()
r.extract_keywords_from_text(new_message)
keywords = r.get_ranked_phrases()  # Returns keywords with highest rank first

# Find synonyms for each keyword and construct a faux-narrative
synonyms = []
for keyword in keywords[:5]:  # Adjust the number of keywords as needed
    synsets = wordnet.synsets(keyword)
    # Add the first synonym for each keyword to the faux-narrative
    if synsets:
        synonyms.append(synsets[0].lemmas()[0].name())
faux_narrative = ', '.join(synonyms)

# Construct the logic-based output
output = [{"input": keyword, "output": synonym} for keyword, synonym in zip(keywords, synonyms)]

# Use the OpenAI API to generate a response to the modified message
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a DeNarrator, an assistant designed to understand human inputs, strip them of narrative elements, and present them in a non-narrative, data-focused format."},
        {"role": "user", "content": faux_narrative},
    ]
)

# Print the AI's response in JSON format
print(json.dumps(response['choices'][0]['message'], indent=4))

# Print the logic-based output
print(json.dumps(output, indent=4))
