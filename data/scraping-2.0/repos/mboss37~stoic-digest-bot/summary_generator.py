import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_prompt_with_letter(letter_content):
    with open("prompt.json", "r") as file:
        prompt_data = json.load(file)
    
    # Replace placeholder with actual letter content
    prompt_data[1]['content'] = prompt_data[1]['content'].replace('{letter_content}', letter_content)
    
    return prompt_data

def generate_summary(letter_content):
    summary = ""
    try:
        messages = load_prompt_with_letter(letter_content)

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )

        for chunk in completion:
            if "content" in chunk.choices[0].delta:
                delta_text = chunk.choices[0].delta["content"]
                summary += delta_text
            
    except openai.error.OpenAIError as e:
        print(f"OpenAI API call failed with the error: {e}")
        return None
    
    return summary.strip()
