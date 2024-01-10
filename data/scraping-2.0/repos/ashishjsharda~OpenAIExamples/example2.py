import openai

#you can use any oen of [davinci,babbage,curie,ada] models
def query_gpt(prompt, model="text-curie-001", max_tokens=10):
   
    
    openai.api_key = 'your open api key'

    # Making a request to the model
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens
    )

    
    return response.choices[0].text.strip()


prompt = "Translate the following English text to French: 'Hello, how are you?'"
response = query_gpt(prompt)
print(response)
