"""
This scratch file explores the idea of having LLMs specifically generate content to help with learning APIs
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
# model = 'gpt-4-1106-preview'
model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
def main():
    db = get_direct_db()
    highlight_source_id = '0a16c1a1-33b3-4777-8d2a-59347d1a985a'
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
        "content": "Nice. What are the key APIs in this subject? Provide a list containing the names of 5 key API classes or methods and their definition/use. Be concise."
    })
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
            "content": "Please convert this list of term/definition pairs to a json object that looks like this:\n{\"results\": [" +
                       "{\"term\": \"term1\", \"definition\": \"definition1\"}]}"
        })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "Nice. Now, please extend this list with 5 more API classes or methods and their definition/use. Be concise."
    })
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
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
            "role": "user",
            "content": "What is an example use-case for these APIs? Be extremely concise."
    })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
            "role": "user",
            "content": "Provide sample code that uses these APIs to implement a use-case you described. Don't bother with import statements. Just the necessary code for the use-case.   Be concise."
    })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("----")
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
            "role": "user",
            "content": "That's great, can you give me another example of code using these APIs in a common use case? Be concise."
    })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("----")

if __name__ == '__main__':
    main()