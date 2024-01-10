import time
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"

def generate_manga_cot(prompt, model_engine="text-davinci-002"):
    # Generate manga content using the OpenAI API
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message.strip()

def three_hop_manga_cot(manga_dialogue):
    # First hop: Determine the main theme of the manga dialogue
    theme_prompt = f"Given the following manga dialogue: '{manga_dialogue}'. What is the main theme?"
    theme = generate_manga_cot(theme_prompt)

    # Second hop: Extract the opinion expressed in the dialogue
    opinion_prompt = f"The main theme of the manga dialogue '{manga_dialogue}' is {theme}. What is the opinion expressed in the dialogue?"
    opinion = generate_manga_cot(opinion_prompt)

    # Third hop: Predict the future development of the story
    future_prompt = f"The main theme of the manga dialogue '{manga_dialogue}' is {theme} and the opinion expressed is {opinion}. What could be the future development of the story?"
    future = generate_manga_cot(future_prompt)

    return {"dialogue": manga_dialogue, "theme": theme, "opinion": opinion, "future": future}

with open('manga4/learning_no_3points_test', 'r', encoding='utf-8') as f:
    with open('learning_no_3points_7.txt', 'w', encoding='utf-8') as out:
        for line in f:
            # manga_dialogue should be set to the current manga dialogue line
            manga_dialogue = line.strip()
            time.sleep(1)  # Add a delay between API calls to avoid rate limits
            result = three_hop_manga_cot(manga_dialogue)
            print(result)
            out.write(result.__str__() + "\n")
            out.flush()
