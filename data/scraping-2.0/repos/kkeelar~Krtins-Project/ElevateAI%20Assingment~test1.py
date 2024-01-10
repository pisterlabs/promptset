import openai

openai.api_key = ''

try:
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt="Summarize this text:\n\n" + "Your test text here",
        max_tokens=150
    )
    print(response.choices[0].text.strip())
except Exception as e:
    print('Error:', e)
