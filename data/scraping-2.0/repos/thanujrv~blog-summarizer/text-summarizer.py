from newspaper import Article
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()

    return article.text.replace("\n", "")

def prepare_summary_prompt(text):
    tag = "Summarize this for a second-grade student:\n"

    return tag + text + "\n\n"

def summarize_text(text_data):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prepare_summary_prompt(text_data),
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response.choices[0].text

