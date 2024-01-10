import openai

openai.api_key = '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'
def get_completion(prompt, model = "gpt-3.5-turbo"):
    messagens = [{"role" : "user", "content" : prompt}]
    response = openai.ChatCompletion.create(
        model = model, 
        messagens = messagens,
        temperature = 0,
        )
    return response.choices[0].message["content"]

prompt = "quando foi a primeira guerra mundial"
response = get_completion(prompt)
print(response)