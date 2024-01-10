import openai

def askGPT(text):
    # openai.api_key = "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"

    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = text,
        temperature = 0.6,
        max_tokens = 150,
    )
    return print(response.choices[0].text)

def main():
    while True:
        print('GPT: Ask me a question\n')
        myQn = input()
        askGPT(myQn)

main()