'''
This program is for summarizing large amounts of text using GPT-3.

It takes a filename on the command line and reads the text from that file.

Then it allows the user to interactively summarize the text.
'''

import setcreds
import openai
import sys

def main():
    # read from stdin and write to stdout

    # read the file
    lines = sys.stdin.readlines()

    # remove every even numbered line
    lines = [line for i, line in enumerate(lines) if i % 2 == 1]

    # remove trailing newlines
    lines = [line.rstrip() for line in lines]

    # write the lines to stdout
    sys.stdout.write("\n".join(lines))


# call main() if this is the main module
if __name__ == "__main__":
    main()
