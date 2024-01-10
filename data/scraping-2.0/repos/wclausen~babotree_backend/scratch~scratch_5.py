"""
This scratch file is for experimenting with the effectiveness of LLMs at
extracting key points from highlights
"""
import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, HighlightSource

openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)
def main():
    db = get_direct_db()
    highlight_source_id = '9664c4d7-9735-4e3c-a42e-cea68ca4ed37'
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id).all()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + "\n".join(
                [highlight.text for highlight in docker_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "Please identify a list of key ideas from the excerpts. Return your response as markdown. Be concise. Take a deep breath. Reason through the task step-by-step."
        }
    ]
    response = openai_client.chat.completions.create(
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    # now make it more concise, as always


if __name__ == '__main__':
    main()