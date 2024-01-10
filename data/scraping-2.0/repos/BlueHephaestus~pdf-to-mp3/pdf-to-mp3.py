import sys, pathlib, fitz
from pathlib import Path
from openai import OpenAI

fname = sys.argv[1]  # get document filename
if len(sys.argv) > 2:
    outname = sys.argv[2]  # get output filename
else:
    outname = fname

# if len(sys.argv) > 3:
#     model = sys.argv[3]  # get model name
# else:
#     model = "tts-1" #default

if len(sys.argv) > 3:
    model = "tts-1-hd"  # get model name
else:
    model = "tts-1" #default

with fitz.open(fname) as doc:  # open document
    pages = [[word[4] for word in page.get_text("words")] for page in doc]

text = []
# Flatten
for page in pages:
    text += page

# Remove citations
import re

def remove_citations(s):
    # This regular expression matches:
    # - An opening square bracket '['
    # - Followed by one or more digits '\d+'
    # - Followed by zero or more groups of:
    #   - A comma, possibly surrounded by whitespace '\s*,\s*'
    #   - Followed by one or more digits '\d+'
    # - Followed by a closing square bracket ']'
    pattern = r'\[\d+(?:\s*,\s*\d+)*\]'
    return re.sub(pattern, '', s)

# Now have list of words in the file
client = OpenAI(api_key=open(Path(__file__).parent / "apikey.txt").read().strip())

#CHUNK_SIZE= 100#words
INPUT_SIZE=4096#chars

from tqdm import tqdm

def chunk_iter(text):
    # Iterate through the max size each input window can be, clipping on word boundaries.
    chunk = ""
    for word in text:
        if len(chunk+word) < INPUT_SIZE-1:#-1 for space
            chunk += word + " "
        else:
            yield remove_citations(chunk) # doesn't work over word boundaries but this is a QOL thing
            chunk = word + " "

speech_file_path = Path(__file__).parent / f"{outname}.mp3"
open(speech_file_path, "w").close() # clear the file
for chunk in tqdm(chunk_iter(text), total=len(' '.join(text))//INPUT_SIZE):
    response = client.audio.speech.create(
      model="tts-1-hd",
      voice="fable",
      input=chunk,
    )
    # add this chunk to the file
    with open(speech_file_path, "ab") as f:
        for chunk in response.iter_bytes(chunk_size=1024):
            f.write(chunk)


