import os
import openai
import sqlite3

GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY")

conn = sqlite3.connect("database.db")

response = openai.ChatCompletion.create(
    model=GPT_MODEL,
    messages=[{"role:": "user", "content": "Hello, Who are you?"}],
)
response
