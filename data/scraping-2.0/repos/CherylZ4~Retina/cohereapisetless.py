import cohere
import keys
def grabDefinition(word: str):
    co = cohere.Client(keys.cohereapikey)
    response = co.generate(
    model='command-xlarge-nightly',
    prompt=f'What is {word}?',
    max_tokens=50,
    temperature=0.8,
    stop_sequences=["--"],
  )
    return (response.generations[0].text.strip('--').strip('\n').lower())