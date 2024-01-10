import openai


def getText(prompt, tokens):
    openai.api_key = "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" #disabled key

    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=tokens)
    # print(response.get('choices'))
    print(response['choices'][0]['text'])
    textResponse = response['choices'][0]['text']
    return textResponse
