import openai

FINE_TUNED_MODEL = "ada:ft-ktinformatik-2023-05-02-07-45-24"
YOUR_PROMPT = "Translate from English to Bengali: ghurte"

# Print the prompt
print("Prompt:", YOUR_PROMPT)

# Call the OpenAI API with the fine-tuned model and the prompt
response = openai.Completion.create(
    model=FINE_TUNED_MODEL,
    prompt=YOUR_PROMPT
)

# Extract the translated text from the response
translated_text = response.choices[0].text.strip()

# Print the translated text
print("Translated text:", translated_text)
