"""
This scratch file explores the idea of having LLMs "extend" the highlights that we already have, to fill in gaps, or
push knowledge forward
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
    highlight_source_id = '5269a1c0-1774-4f25-ab7c-a41554c2a0b4'
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
        "content": "Nice. What are the big ideas about this topic, both those in the excerpts and including other knowledge you have? Be concise."
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
        "content": "Excellent. Can you now describe how these big ideas are connected to each other? Be concise."
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
    # messages.append({
    #     "role": "assistant",
    #     "content": response.choices[0].message.content
    # })
    # messages.append({
    #     "role": "user",
    #     "content": "Please give me 3 question/answer pairs about these expert-level topics. Be concise, but make sure the questions are deep. Take a deep breath."
    # }
    # )
    # # mistralai/Mixtral-8x7B-Instruct-v0.1
    # response = openai_client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=1.0,
    #     max_tokens=1000,
    #     top_p=1.0,
    #     frequency_penalty=.25,
    #     presence_penalty=.25,
    # )
    # print(response.choices[0].message.content)
    # print("-----")
    # and another one
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "That's really great. Can you provide a list of term/definition pairs for the new terms/definitions you've referenced here?"
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
    # let's try to take this last answer and get the llm to convert it to a json object in a new convo
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": response.choices[0].message.content
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "Please convert this list of term/definition pairs to a json object that looks like this:\n{\"results\": [" +
                       "{\"term\": \"term1\", \"definition\": \"definition1\"}]}"
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




if __name__ == '__main__':
    main()