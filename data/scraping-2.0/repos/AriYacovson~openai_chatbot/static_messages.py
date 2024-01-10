import dotenv
import openai

# Configure OpenAI API key
openai.api_key = dotenv.get_key(".env", "OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{
        'role': 'system',
        'content': 'You are a helpful assistant.'}, {
        'role': 'user',
        'content': 'Write me a 3 paragraph bio'
    }],
    temperature=0.6
)
print(response['choices'][0]['message']['content'])
