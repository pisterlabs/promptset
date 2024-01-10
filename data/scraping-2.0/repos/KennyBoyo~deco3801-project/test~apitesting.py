import openai

#dont leak this pls
openai.api_key = "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"

prompt = "Write a wikipedia article about the University of Queensland."

output = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=1000,
    temperature=0.1
)

#print(type(output["choices"][0]))
print("Output:", output["choices"][0]["text"])
print("Total Tokens:", output["usage"]["total_tokens"])