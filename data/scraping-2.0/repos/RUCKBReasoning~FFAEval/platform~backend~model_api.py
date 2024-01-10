from openai import OpenAI

def get_stream_response(model, api_key, base_url, messages):
    client = OpenAI(api_key=api_key, base_url=base_url,)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": ("user" if i % 2 == 0 else "assistant"),
                "content": messages[i]
            } for i in range(len(messages))
        ],
        stream=True,
    )
    return stream
