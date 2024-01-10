import openai

openai.api_key = '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'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,
    )
    
    return response.choices[0].message["content"]


prompt = "WHO IS THE PRESIDENT OF UNITED STATES"
response = get_completion(prompt)

print(response)