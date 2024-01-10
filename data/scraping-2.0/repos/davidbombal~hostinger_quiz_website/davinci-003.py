import openai

API_KEY = "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"
openai.api_key = API_KEY

# This is the latest model that is available to the public. 
# There is a beta for GPT-4 open to a closed group of developers.
model = "text-davinci-003"

res = openai.Completion.create(
    prompt="Create 20 CCNA exam questions. It must be JSON formatted with question, option a, option b, option c, option d, correct answer, explanation.",
    model=model,
    # Number of tokens to generate
    max_tokens=4061,
    # Number of unique responses. Range of 0 - 1
    # 0 means no unique response. 
    # 1 means every response is unique.
    temperature=0.5,
    # Number of outputs
    n=1,
)

with open("ccna1.json", "w") as f:
    f.write(res["choices"][0]["text"])
