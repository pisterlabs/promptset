from text_processor import tokenize_text, remove_stopwords

def generate_esg_report(env_content, soc_content, gov_content, openai_api_key):
    env_tokens = remove_stopwords(tokenize_text(env_content))
    soc_tokens = remove_stopwords(tokenize_text(soc_content))
    gov_tokens = remove_stopwords(tokenize_text(gov_content))

    all_tokens = env_tokens + soc_tokens + gov_tokens
    input_text = ' '.join(all_tokens)

   
    import openai
    openai.api_key = openai_api_key

    prompt = f"Generate an ESG compliance report based on the following input:\n\n{input_text}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=32000
    )

    report_text = response['choices'][0]['text']
    return report_text
