import openai
openai.api_key = "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"

model = "gpt-3.5-turbo"
#prompt = "You are an expert cook who cares about sustainability, and you want to help the user utilize all of the ingredients that they currently possess. Currently, they have 2 chicken breasts, 4 eggs, 1 pound of flour, vegetables like onions and tomatoes and garlic, and butter. You can assume they have other basic ingredients like salt, oil, pepper, etc. Recommend a tasty meal that the user can make while providing a recipe for the meal using exact measurements. Feel free to suggest the user to buy some items."
response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)

generated_text = response.choices[0].text
print(generated_text)