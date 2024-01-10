import openai
from tqdm import tqdm

with open("/Users/jasper/oai.txt", 'r') as f:
    openai.api_key = f.read()

prompt = "Return a bunch of random semantically valid, sensical sentence of your choosing. Each new sentence should be on a new line. Do not put any numbers at the beginning of the sentence."

for _ in tqdm(range(10)):
    sentences = []
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=1,
    max_tokens=1000,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    )

    sentence = completion.choices[0].message.content
    each_sentence = sentence.split('\n')
    for s in each_sentence:
        if len(s) > 0:
            sentences.append(s)

    # append the new sentences to sensical_sentences.txt
    with open("sensical_sentences.txt", 'a') as f:
        for sentence in sentences:
            f.write(sentence.strip() + '\n')