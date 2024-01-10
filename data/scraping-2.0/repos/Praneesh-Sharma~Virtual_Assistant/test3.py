import openai
openai.api_key = "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"


def chatgpt(user_query):
    response = openai.Completion.create(engine='text-ada-001',
                                        prompt=user_query,
                                        n=1,
                                        temperature=0.5)
    return response.choices[0].text


prompt = "Hello, how are you?"

print(chatgpt(prompt))
