import openai
from prompt import Prompt
import argparse
import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser(description="audio2md")
parser.add_argument("-f", "--file", help="File path", required=True)
parser.add_argument("-p", "--prompt", help="Prompt", required=False, default=None)
parser.add_argument("-r", "--refine", help="Refine prompt", required=False, default=False)
parser.add_argument("-o", "--output", help="Output file", required=False, default='output')
args = parser.parse_args()

OUTPUT_DIR = 'outputs'
audio_file = args.file
prompt = args.prompt
refine = args.refine
output = args.output


prompt_template = Prompt(prompt, refine=refine)

print('Transcribing audio...')
transcription = openai.Audio.transcribe('whisper-1', open(audio_file, "rb"))

print('Generating markdown...')
md_reponse = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": f"{prompt_template.prompt}"},
        {"role": "user", "content": f'{transcription["text"]}'}
    ],
)

print('Writing markdown...')
with open(f'{OUTPUT_DIR}/{output}.md', 'w') as f:
    f.write(md_reponse.choices[0]['message']['content'])
    
    