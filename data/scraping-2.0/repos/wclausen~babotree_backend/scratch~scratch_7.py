import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight

together_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)

openai_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('OPENAI_API_KEY'),
)

openai_client = together_openai_client
# model = 'gpt-4-1106-preview'
model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
def main():
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == '9f1988ce-acd3-4c06-88d7-fa380769f0ca').all()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Here are some excerpts from a source text called \"Docker Deep Dive\" :\n" + "\n".join(
                [highlight.text for highlight in docker_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "Identify the key concepts in the excerpts above and create a MermaidJS diagram connecting the concepts. Be sure to label the connections between nodes. Be concise."
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



if __name__ == '__main__':
    main()