import os
import sys

import openai
import argparse

from copy import copy
from dotenv import load_dotenv


def load_file(input_file_path):
    transcript_path = os.path.join(input_file_path)
    with open(transcript_path , "r") as text_file:
        transcript_text = text_file.read()
    return transcript_text

def chunk_and_query(output_file_name, output_folder_name, prepend_query, transcript_text):

    every_x_chars = 5000
    num_transcriptions = int(len(transcript_text)/every_x_chars)+1
    print(f"Creating {num_transcriptions} sub summaries...")
    for transcript_chunk in range(num_transcriptions):
        if transcript_chunk == num_transcriptions-1:
            user_query = prepend_query +  transcript_text[(transcript_chunk)*every_x_chars:]
            if len(user_query)<=500:
                continue
        else:
            user_query = prepend_query +  transcript_text[transcript_chunk*every_x_chars:(transcript_chunk+1)*every_x_chars]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role":"user", "content":user_query}],
            #     functions=function_descriptions,
            #     function_call='auto'
        )

        message = response['choices'][0]['message']
        
        sub_transcript_file_path = os.path.join(output_folder_name, output_file_name).replace('.txt',f'_{transcript_chunk}.txt')
        
        with open(sub_transcript_file_path , "w") as text_file:
            text_file.write(message.content)

    print("Creating meta summary...")
    to_summarize = ''
    for i in range(num_transcriptions):
        sub_transcript_file_path = os.path.join(output_folder_name, output_file_name).replace('.txt',f'_{i}.txt')
        if not os.path.exists(sub_transcript_file_path):
            continue
        with open(sub_transcript_file_path , "r") as text_file:
            sub_transcript = text_file.read()
        if i == 0:
            to_summarize += copy(sub_transcript)
        else:
            to_summarize =  to_summarize + ' transcript break '
            to_summarize =  to_summarize +  copy(sub_transcript)

    print(f"Summary character length: {len(to_summarize)}, token char limit somewhere around 12-15K")
    prepend = 'The following is a series of broken, piecewise podcast summaries delimited by transcript break. Can you combine all of them into a single summary, extracting as much unique information as possible? '
    prepend += to_summarize
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[{"role":"user", "content":prepend}],
        #     functions=function_descriptions,
        #     function_call='auto'
    )

    message = response['choices'][0]['message']
    with open(os.path.join(output_folder_name, output_file_name) , "w") as text_file:
        text_file.write(message.content)
    print("Done! File written to {output_folder_name}/{output_file_name}}")

    for i in range(num_transcriptions):
        os.remove(os.path.join(output_folder_name, output_file_name).replace('.txt',f'_{i}.txt'))
    return

def main():
    env = load_dotenv()
    openai.api_key = os.getenv("OPEN_API_KEY")

    parser = argparse.ArgumentParser(description='Summarize a transcription text file')
    parser.add_argument('--input_file', type=str, help='Input file to summarize')
    parser.add_argument('--output_file', type=str, help='Output file to dump summary')
    parser.add_argument('--output_folder', type=str, help='Folder to dump output file')
    parser.add_argument('--prepend_query', type=str, help='Prepend query to summary')

    args = parser.parse_args()

    transcript = load_file(args.input_file)
    chunk_and_query(args.output_file, args.output_folder, args.prepend_query, transcript)


    return


if __name__ == "__main__":
    main()






