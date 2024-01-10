import time
import os
import warnings
import openai
from dotenv import load_dotenv

from .utils import get_caption_folder, get_processed_caption_folder, get_transcript, CHANNEL

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
TOKEN_SIZE = 3
MAX_TOKENS_PER_REQ = 2000

INSTRUCTION = """
Split text into proper paragraphs. 
Paragraphs should be at most 5-6 sentences long.
DONT change the text.
Dont add or omit any words.
Dont split sentences in the middle
Add proper punctuation.
"""

def slice_for_tokens(name, text) -> list[str]:
    sentences = text.split("\n")
    if len(sentences) == 1:
        return [text]

    res = []
    sentence_index = 0
    while sentence_index < len(sentences):
        tmp = []
        total_length = 0
        while sentence_index < len(sentences) and total_length + len(sentences[sentence_index]) < (MAX_TOKENS_PER_REQ * TOKEN_SIZE):
            tmp.append(sentences[sentence_index])
            total_length += len(sentences[sentence_index])
            sentence_index += 1
        res.append(' '.join(tmp))

    print(f"{name}: {len(sentences)} sentences were fit into {len(res)} chunks")
    return res

def slice_all_transcripts(sliced):
    res = {}
    for f in sliced:
        p = os.path.join(file_path, f)
        if os.path.isdir(p):
            continue
        already_processed = os.path.exists(os.path.join(get_processed_caption_folder(CHANNEL), "par_" + f))
        if already_processed:
            print(f"Skipping {f} because it was already processed")
            continue
        cur_transcript = get_transcript(p)
        sliced = slice_for_tokens(f, cur_transcript)
        # had_zeros = any([bool(len(s) == 0) for s in sliced])
        # print([bool(s) for s in sliced])
        # if had_zeros:
        # print(f"Check tokenization for {f}", [len(s) for s in sliced])
        res[f] = sliced
    return res

def process_sliced_transcript(sliced_transcript: list[str]):
    processed = []
    for i, piece in enumerate(sliced_transcript):
        print(f"Processing piece {i} of {len(sliced_transcript)}")
        print(piece[:100] + "..." + piece[-100:])
        
        chat_history = [
            {"role": "system", "content": "You are a helpful tool that formats text into paragraphs."},
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": piece}
        ]
        # chat_history.append(chunk)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history
            
            # input=piece,
            # n=1,
            # instruction=INSTRUCTION
        )
        reason = response['choices'][0]['finish_reason']
        message = response['choices'][0]['message']['content']

        print(f"Received result for piece {i}")
        print(message[:100] + "..." + message[-100:])
        if reason != 'stop':
            warnings.warn(f"Finished with code {reason} for")

        processed.append(message)
        time.sleep(5)

    return '\n\n'.join(processed)


print("Loading captions")

file_path = get_caption_folder(CHANNEL)
caption_files = os.listdir(file_path)
sliced_transcipts = slice_all_transcripts(caption_files)

print("Loaded captions")

for name, transcript in sliced_transcipts.items():
    print("Processing", name)
    paragraphs = process_sliced_transcript(transcript)
    time.sleep(30)
    file_path = get_processed_caption_folder(CHANNEL)
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, "par_" + name), "w", encoding="utf-8") as f:
        f.write(paragraphs)
        f.close()
        print("Wrote", name)
    
