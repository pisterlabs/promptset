import openai


def get_summary(article_text, summary_type='one-sentence'):
    if summary_type == 'tldr':
        suffix = "\n\ntl;dr:"
    elif summary_type == 'one-sentence':
        suffix = '\nOne-sentence summary:'
    response = openai.Completion.create(
        engine="curie",
        prompt=article_text + suffix,
        temperature=0.5,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[".", "!", "?"]
    )
    return response['choices'][0]['text']
