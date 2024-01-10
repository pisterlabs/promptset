from _openai.config import API_KEY, get_model_by_id, debit_request_from_model
def call(model_id, content):
    import openai
    openai.api_key = API_KEY
    status, data = debit_request_from_model(model_id)
    if status != 0:
        raise Exception('This model is not eligible for any further requests\nIssue a new model if needed')

    model = get_model_by_id(model_id)
    try:
        response = openai.Completion.create(
            engine= model['name'],
            temperature= model['temperature'],
            max_tokens= model['max_tokens'],
            top_p= model['top_p'],
            frequency_penalty= model['frequency_penalty'],
            presence_penalty= model['presence_penalty'],
            prompt= content
        )
    except Exception as e:
        model = get_model_by_id(model_id, 1)
        raise e
    return response
