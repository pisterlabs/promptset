import os
import re
import json
import openai

from pathlib                    import  Path
from itertools                  import  chain
from datetime                   import  datetime
from dotenv                     import  load_dotenv
from pdfminer.high_level        import  extract_text

from representative             import  Representative
from interviewer                import  Interviewer
from extract                    import  *
from utilities                  import  *

def formatted_time() -> str:
    now =                       datetime.now()
    formatted_time =            now.strftime('%H-%M-%d-%m-%Y')
    return formatted_time

def format_subtopics_with_quotes(subtopics: list[str]) -> str:
    if (len(subtopics) == 0):
        return  "<|no subtopics|>"
    if (len(subtopics) == 1):
        return  subtopics[0]
    formatted_string = ""
    for subtopic in subtopics[:-1]:
       formatted_string +=      f"\"{subtopic}\", " 
    formatted_string +=         f"and \"{subtopics[-1]}\""
    return formatted_string

def format_subtopics(subtopics: list[str]) -> str:
    if (len(subtopics) == 0):
        return  "<|no subtopics|>"
    if (len(subtopics) == 1):
        return  subtopics[0]
    formatted_string =          ""
    for subtopic in subtopics[:-1]:
       formatted_string +=      f"{subtopic}, " 
    formatted_string +=         f"and {subtopics[-1]}"
    return  formatted_string

if __name__ == "__main__":
    subtopics =     ["one", 'two', 'three']
    print(format_subtopics(subtopics))