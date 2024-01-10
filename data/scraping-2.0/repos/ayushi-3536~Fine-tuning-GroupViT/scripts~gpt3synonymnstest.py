import openai

def generate_synonyms(word):
    prompt = f"Generate visually meaningful synonyms for the word '{word}':"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=5,  # Generate 5 synonyms
        stop=None,
        temperature=0.8
    )

    synonyms = [choice["text"].strip() for choice in response.choices]
    return synonyms

# Set up OpenAI API credentials
openai.api_key = "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"

# Example usage
word = "car"
synonyms = generate_synonyms(word)

print(f"Synonyms for '{word}':")
for synonym in synonyms:
    print(synonym)
