import openai # pip install openai
import sys


openai.api_key = "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"

prompt = str(sys.argv[1])

print(prompt)

request = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = prompt,
    max_tokens = 1024,
    n = 1,
    stop = None,
    temperature = 0.5,
)

response = request.choices[0].text
print(response)