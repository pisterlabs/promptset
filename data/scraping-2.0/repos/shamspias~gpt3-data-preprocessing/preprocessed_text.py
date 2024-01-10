import openai
import re

# Load OpenAI GPT-3 API
openai.api_key = "YOUR_API_KEY"
model_engine = "text-davinci-003"


# Read article text
with open("article.txt", encoding='utf-8') as f:
    text = f.read()

# Split text into sentences
sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)

# Create input-output pairs
pairs = []
for i in range(len(sentences) - 1):
    pairs.append([sentences[i], sentences[i+1]])

# Tokenize and encode input-output pairs
encoded_pairs = []

for pair in pairs:
    prompt = pair[0]
    response = pair[1]

    prompt_completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response_completion = openai.Completion.create(
        engine=model_engine,
        prompt=response,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    prompt_tokens = prompt_completion.choices[0].text
    response_tokens = response_completion.choices[0].text

    encoded_pairs.append([prompt_tokens, response_tokens])

# Save encoded pairs to file
with open("encoded_pairs.txt", "w") as f:
    f.write(str(encoded_pairs))
