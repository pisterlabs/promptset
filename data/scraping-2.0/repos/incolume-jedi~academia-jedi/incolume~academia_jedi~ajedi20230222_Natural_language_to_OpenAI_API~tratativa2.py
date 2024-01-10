import openai
from config import settings

openai.api_key = settings.OPENAI_API_KEY

response = openai.Completion.create(
    model='text-davinci-003',
    prompt="hi\n\nI'm an AI bot. I don't understand your message. Please rephrase it.",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
