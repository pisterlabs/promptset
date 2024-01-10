import openai
openai.api_key = "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"

def gpt_engine(PROMPT, MaxToken=50, output=3):
    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = PROMPT,
        max_tokens = MaxToken,
        n=output
    )

    output = list()
    for res in response['choices']:
        output.append(res['text'].strip())
    return output


p = '''Write a poem for busy life of and indian engineering college student's life, their struggles, fun, mistakes, hardwork '''

result = gpt_engine(p)
print(result)