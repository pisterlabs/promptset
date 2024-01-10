#!/usr/bin/env python

import openai
import sys

def console_print(st=u"", f=sys.stdout, linebreak=True):
    #global enc
    #assert type(st) is unicode
    #f.write(st.encode(enc))*/
    f.write(st)
    if linebreak: f.write(os.linesep)

def main(argv) :
    audio_file = open(argv[0], "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    console_print(transcript)

if __name__ == "__main__":
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
