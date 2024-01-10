import openai
import os
import re


def read_transcript(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def merge_author_lines(text):
    
    lines = text.split("\n")
    
    #add ":" to the end of odd lines
    for i in range(0, len(lines), 2):
        lines[i] = lines[i] + ":"
        
    i = 0
    skip=0
    while i < len(lines) - skip-2:
        print('-------------------')
        print(i, lines[i], skip, lines[i + 2+skip])
        if lines[i] == lines[i + 2+skip]:
            del lines[i + 2]
            skip +=1
        else:
            i += 2+skip
            skip=0

    return "\n".join(lines)


def transform_text(text):
    # Remove lines with timestamps using a regular expression
    text_no_timestamps = re.sub(r'\d{1,2}:\d{1,2}:\d{1,2}\.\d{1,3} --> \d{1,2}:\d{1,2}:\d{1,2}\.\d{1,3}', '', text)

    # Remove extra newlines and spaces
    cleaned_text = re.sub(r'\n\s*\n', '\n', text_no_timestamps).strip()

    # Combine speaker with their dialogue in the same line
    combined_text =merge_author_lines(cleaned_text)

    return combined_text


# File path to the transcript
file_path = 'transcript_raw.txt'

trasformed_text = transform_text(read_transcript(file_path))
print (trasformed_text)


#save the meeting minutes to a file
with open('data\\transcript.txt', 'w') as file:
    file.write(trasformed_text)
    
