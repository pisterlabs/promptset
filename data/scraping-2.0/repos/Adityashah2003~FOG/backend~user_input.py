import openai
import keywords

def generate_response(prompt, model="text-davinci-003", max_tokens=50):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return response.choices[0].text.strip()

def extract_kw(user_message):
    user_message = f"Extract fashion outfits items from these: {user_message}"
    input_kw = generate_response(user_message)
    return input_kw

def main(user_message):
    input_kw = extract_kw(user_message)
    final_data = keywords.main(input_kw)
    return final_data
