import openai

# Set up the API credentials
openai.api_key = "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"


def get_response(p: str):
    query = f'{p} summary'
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        temperature=0.0,
        max_tokens=100
    )
    return response


def fc(major: str, univ1: str, univ2: str, univ3: str)\
        -> tuple[str, str, str, str]:
    r_major = get_response(major)
    r_univ1 = get_response(univ1)
    r_univ2 = get_response(univ2)
    r_univ3 = get_response(univ3)

    generated_text_major = r_major.choices[0].text.strip()
    generated_text_univ1 = r_univ1.choices[0].text.strip()
    generated_text_univ2 = r_univ2.choices[0].text.strip()
    generated_text_univ3 = r_univ3.choices[0].text.strip()

    return generated_text_major, generated_text_univ1, generated_text_univ2, \
        generated_text_univ3

