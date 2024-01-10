import openai

gpt_model = 'gpt-3.5-turbo'

def json_cleaner(raw_txt): 

    system_message = '''
    you are responsible for isolating and cleaning JSON.
    e.g.:
    '["item_1", "item_2"]'
    Make sure that you only respond with the cleaned JSON, and nothing else.
    it is ok to not change the input if it is already correct
    '''

    messages = [
            {"role":"system",
            "content":system_message},
            {"role":"user",
            "content":f'please fix this JSON: {raw_txt}. Do not respond with anything except the fixed JSON'}
        ]

    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages
    )

    response_dict = response.to_dict()
    raw_text = response_dict['choices'][0]['message']['content']

    return raw_text