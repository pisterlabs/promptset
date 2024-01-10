import openai
import os
from dotenv import load_dotenv
load_dotenv()

def translate_batch(batch):
    """Translate a batch of text lines into Spanish using OpenAI."""
    prompt = "I am sending you a list of available essential oils from a trading house. I would like to translate this list into Spanish. Can you output a list that uses tab stops as column separators and has the same content, only translated into Spanish? The Latin names should not be translated, only reproduced. I only need the columms 'Code, Name, Lat. name, Origin, Quality'.\n\n" + "\n".join(batch)
    
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # Extracting the text from the last response message
    last_message = response['choices'][0]['message']['content']
    return last_message.strip()

def process_file():
    """Process the source file and write translations to the target file."""
    source_file = 'source.txt'
    target_file = 'translation.txt'

    with open(source_file, 'r') as source, open(target_file, 'w') as target:
        lines = source.readlines()
        batch_size = 14

        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            translated_batch = translate_batch(batch)
            target.write(translated_batch + '\n\n')

if __name__ == "__main__":
    process_file()
