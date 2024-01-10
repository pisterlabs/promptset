import openai


def get_classification_results(prompt, model_selection, openai_key):
    # creds
    openai.api_key = openai_key

    # get response
    kwargs = {
        'model': model_selection,
        'messages': prompt,
    }
    response = openai.ChatCompletion.create(**kwargs)

    # parse response
    result = response.choices[0].message.content
    return result
