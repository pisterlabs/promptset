from openai import OpenAI

client = OpenAI(api_key='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')


def callGPT(prompt):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    detail = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            detail += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="")

    return detail

# if __name__ == '__main__':
#     print(callGPT("Description tulip"))
