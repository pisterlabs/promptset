import openai
import re

openai.api_key = "your-api-key"

def check_vulgarity(text):
    vulgar_words = ["No"]
    
    for word in vulgar_words:
        if re.search(rf'\b{word}\b', text, re.IGNORECASE):
            return False
    
    return True


def petition_checker(petition, temperature=0.7):

    prompt = (
             f"Please evaluate the following petition and determine whether it aligns with the criteria for validity. Consider aspects such as the promotion of development, prosperity of the nation, and improvements in the educational, health,tourism and agriculture sectors. If you find the petition valid, respond with a single word 'yes'; otherwise, respond with 'no'.\n\n{petition}\n"
    )

    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=800,
            temperature=temperature,
    )

    review_one = response.choices[0].text.strip()
    # Check for vulgarity in the full article
    result_message = check_vulgarity(review_one)
    return result_message
