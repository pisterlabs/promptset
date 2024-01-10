import random

def unknown(message):
    import openai

    # Set up the OpenAI API client
    openai.api_key = "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"

    # Set up the model and prompt
    model_engine = "text-davinci-002"
    prompt = message

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    print(response)
    return response