from openai import OpenAI
import re
#%%
# Set up the API key
#openai.api_key = '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'
# Define the prompt to generate text completion
prompt = "Energy Crisis"
#%%
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='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',
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ],
  n=1,
  max_tokens=256,
  temperature=0.5,
)
#%%

if user_language == 'de':
    with open('german_stopwords_full.txt', 'r') as f:
        german_stopwords = set(f.read().splitlines())
    stop_words = german_stopwords


#%%
# Extract the generated text
generated_text = response.choices[0].message.content

# Split the generated text into words
words = re.findall(r'\w+', generated_text)

# Create a set to store the unique keywords
keywords = set()

# Add each word to the set if it's not a stop word
stop_words = {"the", "and", "of", "to", "in", "that", "is", "with", "for", "on"}
for word in words:
    if word.lower() not in stop_words:
        keywords.add(word.lower())

# Convert the set of keywords to a list
keywords = list(keywords)

# Print the generated keywords
print("Generated keywords:")
for keyword in keywords:
    print("- " + keyword)

