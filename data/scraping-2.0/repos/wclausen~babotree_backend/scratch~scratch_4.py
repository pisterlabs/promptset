import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight

openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)

def main():
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == '0a16c1a1-33b3-4777-8d2a-59347d1a985a').all()
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
            "content": "Develop a graduate-level set of questions/answers based on the excerpts. Focus on advanced topics. The questions/answers do not need to be directly from the excerpts, but they should be focused on the same topic."
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

if __name__ == '__main__':
    main()