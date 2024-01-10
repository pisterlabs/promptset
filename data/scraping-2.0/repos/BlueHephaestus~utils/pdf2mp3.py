"""
Does what it says on the tin.
Given any pdf, will convert it to an mp3.
Will not include tables, will exclude citations like [3] [1,2]
Can be run via python3 pdf2mp3 in.pdf out.mp3 tts-1.
See function doc for more info.

Requires the environment variable OPENAI_API_KEY to be set to a valid key.

TODO can be improved by making the chunks terminate on sentences, not words.
"""
import sys, pathlib, fitz
from pathlib import Path
from openai import OpenAI
import os
import argparse
import re
from ai_utils import tts

# Remove citations
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

def pdf2mp3(input, output, model):
    print("Converting {} to {} using model {}".format(input, output, model))

    with fitz.open(input) as doc:  # open document
        text = chr(12).join([page.get_text() for page in doc])

        # pages = [[word[4] for word in page.get_text("words")] for page in doc]
    #
    # text = []
    # # Flatten
    # for page in pages:
    #     text += page

    text = remove_citations(text)
    tts(text, output, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a pdf to an mp3.')
    parser.add_argument("input", help="Input pdf file")
    parser.add_argument("--output", help="Output mp3 file (defaults to input name)", default=None)
    parser.add_argument("--model", help="Model to use (defaults to tts-1)", default="tts-1")

    args = parser.parse_args()
    if args.output is None:
        # Save as input name in current directory
        args.output = os.path.basename(args.input) + ".mp3"
        args.output = os.path.join(os.getcwd(), args.output)

    pdf2mp3(args.input, args.output, args.model)

