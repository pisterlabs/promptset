# import openai

# openai.api_key = "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"

# def translate_to_Chinese(text):
#     prompt = f"Translate the following English text to Chinese: '{text}'"
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=60
#     )
#     return response.choices[0].text.strip()

# # Example usage
# english_text = "Hello, how are you?"
# french_translation = translate_to_Chinese(english_text)
# print(f"Translated Text: {french_translation}")
