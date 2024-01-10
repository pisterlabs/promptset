import openai

def check_sentence(prompt):
    """
    The check_sentence function takes a string as an argument and returns the same string with any grammatical errors corrected.

    :param prompt: Pass in the sentence to be corrected
    :return: The corrected sentence
    """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f'Correct this to standard English:\n\n {prompt}',
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].text

