from pathlib import Path
import openai

openai.api_key = "EMPTY"  # Not support yet
openai.api_base = ""

model = "vicuna-7b-v1.3"
prompt = """I need to clean up filenames. Those are comic books. I need a format with <series> #tome_number.zip
Comanche-Greg-Hermann-Integrale-NB-T01.zip
Astérix n°03 - Astérix et les Goths.zip
(2021) Elzear (tome 1) - Le dejeuner - Maco [cbz].cbz
Format into JSON.
"""


def main():
    # create a chat completion
    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    # print the completion
    print(completion.choices[0].message.content)


BD_PATH = Path("M:\Bédés")

for path in BD_PATH.rglob("*.zip"):
    print(str(path).removeprefix(f"{BD_PATH}\\"))
