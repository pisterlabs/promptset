import openai


def query_gpt3(prompt, max_tokens=256):
    openai.api_key = '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'

    response = openai.Completion.create(
        model="text-davinci-003",  # Specify GPT-3.5-Turbo model here
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=None,
        n=1
    )

    return response.choices[0].text.strip()


if __name__ == "_main_":
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    result = query_gpt3(prompt)
    print(result)
