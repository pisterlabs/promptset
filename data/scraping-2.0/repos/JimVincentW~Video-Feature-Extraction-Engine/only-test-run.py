import json

from Captions import captions
from json_processing import process_video, write_output_to_file, read_captions_from_file
from openai_chat import generate_openai_chat_response


Path = "/Users/jimvincentwagner/tests/video_1677841652_k9Wj4Wkr2c.mp4"


## Video to Captions
captions = captions(Path)

write_output_to_file("output.json", captions)

## Read Captions from File
captions = read_captions_from_file("output.json")

## Combine Captions into a single string
prompt_lines = "\n".join(captions)

## Generate the response from OpenAI Chat Completion
response = generate_openai_chat_response(prompt_lines)

## Print the response
print(response)