import os
import openai
from dotenv import load_dotenv, dotenv_values
import argparse



def main():

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=input("Enter a prompt: "))

    args = parser.parse_args()       

    user_input = args.input    

    print(f"generating snippet for {user_input}")          
    result = generate_branding_snippet(user_input)
    print(result)

def generate_branding_snippet(prompt: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # subject = "education in chicago"
    enriched_prompt =f"generate upbeat branding snippet for {prompt}"

    response = openai.Completion.create(
        engine="davinci",
        prompt= enriched_prompt,
        temperature=0.7,
        max_tokens=32,
        
        )

    print(response)
    branding_text: str = response["choices"][0]["text"]

    #strip whtiespace
    branding_text=branding_text.strip()

    # add .. to true 
    last_char = branding_text[-1]
    if last_char not in {".", "?", "!"}:
        branding_text += "..."


    # print(branding_text)
    return branding_text


if __name__ == "__main__":
    main()