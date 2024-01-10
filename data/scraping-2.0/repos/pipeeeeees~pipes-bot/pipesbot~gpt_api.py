from pipesbot import creds
import openai

def requestz(message):
    openai.api_key = creds.openai_key
    model_engine = "text-davinci-003"
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
    return response
