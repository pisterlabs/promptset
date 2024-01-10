from flask import jsonify
import openai
def data_preprocessing(text, text_id = ''):
    openai.api_key = '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'

    model = 'text-davinci-003'
    text_to_analyze = text

    completion = openai.Completion.create(
        engine=model,
        prompt=(f"Sentiment analysis of the following text: {text_to_analyze}'\n\nSentiment Score:"),
        max_tokens=1,
        temperature=0
    )
    return {'sentiment': completion.choices[0].text, 'id': text_id}