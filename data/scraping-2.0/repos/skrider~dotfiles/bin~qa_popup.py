import sys
import openai

model = 'text-davinci-003'

fname = sys.argv[1]

# open fname, send to openai, get response, append response to file
with open(fname, 'r') as f:
    text = f.read()

response = openai.Completion.create(
    engine=model,
    prompt=text,
    temperature=0.7,  # specify temperature to control diversity of responses
    max_tokens=60,  # specify maximum number of tokens (words) in the response
    top_p=1,  # specify the value of "top-p" to control the amount of randomness in the response
    frequency_penalty=0,  # specify the frequency penalty to control the balance between relevance and novelty
    presence_penalty=0,  # specify the presence penalty to control the balance between coherence and diversity
)

with open(fname, 'a') as f:
    f.write(response.choices[0].text)

