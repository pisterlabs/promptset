#!/usr/bin/env python3

"""
This script uses the OpenAI GPT-3 API to format or summarize a transcript.

Usage:
    python3 summarizer_gpt3.py -i <input_file> -o <output_file> --prompt <prompt_type>
"""

import argparse
import logging
import os
import sys

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise Exception("Please set the OPENAI_API_KEY environment variable")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def process_transcript(raw_transcript, prompt_type, chunk_size=4000):
#     if prompt_type not in ["format", "summarize"]:
#         raise ValueError("Invalid prompt type. Accepted values are 'format' or 'summarize'.")

#     processed_transcript = ""

#     raw_chunks = [raw_transcript[i:i + chunk_size] for i in range(0, len(raw_transcript), chunk_size)]
#     num_chunks = len(raw_chunks)
#     logger.info("Split transcript into %d chunks", len(raw_chunks))

#     for index, chunk in enumerate(raw_chunks):
#         logger.info("Processing chunk %d of %d...", index + 1, num_chunks)

#         if prompt_type == "format":
#             prompt = f"Please format the following raw transcript for readability, including punctuation, speaker labels (look for semicolons after names), and spacing. Remove filler words:\n\n{chunk}\n"
#             system_message = "You are a helpful assistant that formats raw transcripts for readability."
#         elif prompt_type == "summarize":
#             prompt = f"Please provide a summary of the following transcript. Emphasize brevity and precision of language:\n\n{chunk}\n"
#             system_message = "You are a helpful assistant that summarizes transcripts."

#         try:
#             completion = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": system_message
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 max_tokens=1024,
#                 n=1,
#                 stop=None,
#                 temperature=0.5,
#             )
#             processed_chunk = completion.choices[0].message['content'].strip()
#             processed_transcript += processed_chunk + "\n"
#         except Exception as e:
#             logger.error("Error processing chunk %d: %s", index + 1, str(e))
#             continue

#     return processed_transcript

def process_transcript(raw_transcript, prompt_type, chunk_size=4000):
    if prompt_type not in ["format", "summarize"]:
        raise ValueError("Invalid prompt type. Accepted values are 'format' or 'summarize'.")

    raw_chunks = [raw_transcript[i:i + chunk_size] for i in range(0, len(raw_transcript), chunk_size)]
    num_chunks = len(raw_chunks)
    logger.info("Split transcript into %d chunks", len(raw_chunks))

    summaries = []

    for index, chunk in enumerate(raw_chunks):
        logger.info("Processing chunk %d of %d...", index + 1, num_chunks)

        if prompt_type == "format":
            prompt = f"Please format the following raw transcript for readability, including punctuation, speaker labels (look for semicolons after names), and spacing. Remove filler words:\n\n{chunk}\n"
            system_message = "You are a helpful assistant that formats raw transcripts for readability."
        elif prompt_type == "summarize":
            prompt = f"Please provide a summary of the following transcript. Emphasize brevity and precision of language:\n\n{chunk}\n"
            system_message = "You are a helpful assistant that summarizes transcripts."

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            processed_chunk = completion.choices[0].message['content'].strip()
            if prompt_type == "summarize":
                summaries.append(processed_chunk)
            else:
                summaries.append(processed_chunk + "\n")

        except Exception as e:
            logger.error("Error processing chunk %d: %s", index + 1, str(e))
            continue

    if prompt_type == "summarize":
        combined_summary = ' '.join(summaries)
        summary_prompt = f"Please provide a summary of the following combined summaries. Emphasize brevity and precision of language:\n\n{combined_summary}\n"
        
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            final_summary = completion.choices[0].message['content'].strip()
            return final_summary
        except Exception as e:
            logger.error("Error processing final summary: %s", str(e))
            return ""

    return ''.join(summaries)


def main():
    """
    Main function that reads a raw transcript file, processes it using the selected prompt type, and writes the output to a file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path to the raw transcript file.", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the processed transcript file.",
        default="processed_transcript.txt",
    )
    parser.add_argument(
        "--prompt",
        help="Select a prompt type: 'format' or 'summarize'.",
        default="format",
    )
    args = parser.parse_args()

    raw_transcript_path = args.input

    # ensure output is valid
    if not args.output.endswith(".txt"):
        logger.error("Output file must be a .txt file")
        sys.exit(1)

    # Read from file
    try:
        with open(raw_transcript_path, "r", encoding="utf-8") as myfile:
            raw_transcript = myfile.read()
    except FileNotFoundError:
        logger.error(f"Error: Could not find file {raw_transcript_path}")
        return

    try:
        processed_transcript = process_transcript(raw_transcript, args.prompt)
    except ValueError as e:
        logger.error(str(e))
        return

    print(processed_transcript)

    output_path = args.output

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_transcript)
    except IOError:
        print(f"Error: Could not write to file {output_path}")
        return

    print(f"Processed transcript saved to {output_path}")


if __name__ == "__main__":
    main()
