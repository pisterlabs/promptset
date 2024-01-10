import openai
import os

def set_openai_api_key(api_key):
    openai.api_key = api_key

def generate_openai_story(user_input, folder_path):
    prompt = f"You: {user_input}\nChatbot:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    story_text = response.choices[0].text.strip()

    # Save the generated story to a unique file
    serial_number = len(os.listdir(folder_path)) + 1
    output_file = os.path.join(folder_path, f"story_{serial_number}.txt")
    with open(output_file, "w") as f:
        f.write(story_text)

    return story_text
