"""
This script is for experimenting with getting LLMs to write "expert-level" questions based on the
contents of highlights.
"""
import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, HighlightSource

together_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)

openai_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('OPENAI_API_KEY'),
)

openai_client = together_openai_client
# 'gpt-4-1106-preview'
model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

def main():
    db = get_direct_db()
    highlight_source_id = '9664c4d7-9735-4e3c-a42e-cea68ca4ed37'
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    relevant_highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id).all()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + "\n".join(
                [highlight.text for highlight in relevant_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "What is the topic of the excerpts?"
        }
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # append the response message and ask more questions
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "What is an example of expert-level knowledge about this topic?"
    }
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # and one more time
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "Please give me 3 question/answer pairs about these expert-level topics. Be concise, but make sure the questions are deep. Take a deep breath."
    }
    )
    # mistralai/Mixtral-8x7B-Instruct-v0.1
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # finally, make the question/answer pairs more concise.
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "That's great, can you make the questions and answers more concise?"
    }
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")

if __name__ == '__main__':
    main()