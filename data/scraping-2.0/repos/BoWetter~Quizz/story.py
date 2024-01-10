from openai import OpenAI
import time

def read_adventure():
    try:
        with open('adventure.txt', 'r') as file:
            return file.read()
    except FileNotFoundError:
        print("Ingen fil hittades.")
        return None

def generate_story(adventure_prompt):
    if adventure_prompt:
        client = OpenAI(api_key='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')
        completion = client.completions.create(
            model='text-davinci-003',
            prompt=adventure_prompt,
            max_tokens=200
        )
        return completion.choices[0].text
    else:
        return "Inget äventyr att generera en historia för."

def save_story(story_text):
    with open('latest_story.txt', 'w') as file:
        file.write(story_text)


def main():
    adventure_text = read_adventure()
    if adventure_text:
        generated_text = generate_story(adventure_text)
        save_story(generated_text)
        print("Genererad text:\n", generated_text)

if __name__ == '__main__':
    main()
