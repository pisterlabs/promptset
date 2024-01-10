import openai

openai.api_key = 'YOUR_API_KEY'

text = ""

while len(text) < 50000:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="ある日、森の中で小さなクマが迷子になってしまいました。",
        max_tokens=5000,
        temperature=0.7,
        n = 5,
        stop=None
    )
    
    for choice in response.choices:
        text += choice.text.strip()
        
    print(f"Generated text length: {len(text)}")

text = text[:50000]
print(text)

