import openai

API_KEY = "ENTER YOUR OPENAI TOKEN HERE"
openai.api_key  = API_KEY


def get_gpt3_response(text:str, temperature = 0.6, max_tokens = 50, top_p = 1, frequency_penalty=0.0,
                      presence_penalty=0.0) -> str:

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    return (response['choices'][0]['text'].strip())


def get_gpt3_response_logprobs(input:list, temperature = 0.6, max_tokens = 50, top_p = 1, frequency_penalty=0.0,
                      presence_penalty=0.0):


    response = openai.Completion.create(
        echo = True,
        logprobs = 5,
        engine="text-davinci-002",
        prompt=input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


    logprobs = [(response['choices'][i]['logprobs']['tokens'],response['choices'][i]['logprobs']['token_logprobs'] )
                for i in range(len(input))]
    return logprobs
