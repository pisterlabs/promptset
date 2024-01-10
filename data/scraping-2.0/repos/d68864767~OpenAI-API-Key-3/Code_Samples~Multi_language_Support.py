```python
import openai

def generate_text_in_french(api_key):
    openai.api_key = api_key

    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Écrivez un paragraphe sur l'importance de l'éducation en français.",
      max_tokens=60
    )

    return response.choices[0].text.strip()

def generate_text_in_language(api_key, language, prompt):
    model_name = f"{language}-davinci-002"
    openai.api_key = api_key

    response = openai.Completion.create(
      engine=model_name,
      prompt=prompt,
      max_tokens=60
    )

    return response.choices[0].text.strip()

if __name__ == "__main__":
    api_key = 'your-api-key'  # replace with your actual API key

    # Generate text in French
    french_text = generate_text_in_french(api_key)
    print(f"Generated text in French: {french_text}")

    # Generate text in a specific language
    language = 'spanish'  # replace with the language you want
    prompt = 'Escribe un párrafo sobre la importancia de la educación en español.'  # replace with your prompt in the language you want
    generated_text = generate_text_in_language(api_key, language, prompt)
    print(f"Generated text in {language.capitalize()}: {generated_text}")
```
