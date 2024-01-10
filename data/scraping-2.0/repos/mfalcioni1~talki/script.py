import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_podcast_script(text, duration, api_key):
    openai.api_key = api_key
    system_role = "You are a helper bot that helps people write podcast scripts. You are given a paper and a duration. You must write a podcast script based on the paper for the given duration."
    prompt = (f"Create a podcast script based on the following academic paper for a duration of {duration} minutes: "
              f"{text} \n\n---\n\n"
              "Use the following format for your podcast script:\n\n"
              "Podcast Script:\n"
              "Introduction:\n"
              "{introduction}\n\n"
              "Main Content:\n"
              "{content}\n\n"
              "Closing:\n"
              "{closing}")
    completion = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo-16k',
    messages = [
        {'role': 'system', 'content': system_role},
        {'role': 'user', 'content': prompt}
    ],
    temperature = 0  
    )
    return completion.choices[0].message.content

def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def save_text(text, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', '-p', type=str, default="paper.txt")
    parser.add_argument('--duration', '-d', type=int, default=10)
    parser.add_argument('--save_path', '-s', type=str, default="script.txt")

    args = parser.parse_args()

    paper = load_text_from_file(args.paper)
    script = generate_podcast_script(paper, args.duration, openai.api_key)

    save_text(script, args.save_path)
    

if __name__ == "__main__":
    main()