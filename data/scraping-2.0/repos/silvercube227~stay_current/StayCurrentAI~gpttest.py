import openai,os,sys

openai.api_key = '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'

topic = "politics"
openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "system", "content": "summarize recent news articles about " + topic}
    ]
)