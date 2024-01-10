import openai
key='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'

openai.api_key = key

system = """Your are an AI bot whose job it is to convert the following natural language text into proper LaTeX format. The output should be able to compile and render without error. Think hard about the text and take your time when making the translation, as the text will have many shorthands and abbreviations that need to be interpreted approprately (requests and syntax that comes from the user). Remember to use your knowledge of mathematics to determine the correct syntax for each math expression (such as which values are included in fractions and square roots). Be aware that some mathematical expressions may appear inline and some may be block displays; it is up to you to correctly identify each (block math is rendered between pairs of dollar signs $$...$$ while inline math is rendered with single dollar signs $...$). The purpose of your effort here is to help the user type the least amount of characters/tokens possible while still being able to have the mathematics they intended to typeset render perfectly. Ensure your mathematics is neat (aligning equal signs, including spacing...) and do your best."""

text = """Consider y=x^2. The derivative is 2x"""

# Use fine tuning

def get_response(text):
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:mavreyn::8BrSnZb7",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
    )

    return response['choices'][0]['message']['content']

print(text)
print(get_response(text))