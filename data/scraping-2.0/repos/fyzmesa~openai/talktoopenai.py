import openai
openai.api_key = ""

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-001", #text-davinci-001, text-curie-001. text-babbage-001, text-ada-001
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

##############################################################################

prompt = "Translate in Spanish: My name is Colin"

##############################################################################

response = generate_response(prompt)
print(response)
