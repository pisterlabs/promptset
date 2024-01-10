"""
Process files in an input directory and get embeddings using OpenAI Text Embedding API, writing the embeddings to an output directory.

This script takes an input directory and an output directory as command-line arguments
and processes all the files in the input directory.
For each file, it gets the embedding using the OpenAI Text Embedding API
and writes the embedding to an identically named file in the output directory.
If the line is longer than 8000 tokens, it will split the line into chunks
of <8000 tokens and write each chunk to its own file.
"""

import openai
from openai.embeddings_utils import get_embedding
import sys
import os
import tiktoken
import argparse


def get_embeddings_old(input_file, output_file):
    # Open the input file for reading and the output file for writing
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        # Iterate over the lines in the input file
        for line in input_f:
            # Get the embedding for the line
            embedding = get_embedding(line, engine='text-embedding-ada-002')
            
            # Write the embedding to the output file as a string
            output_f.write(str(embedding) + '\n')

def get_embeddings(input_file, output_file):
    # Open the input file for reading and the output file for writing
    with open(input_file, 'r') as input_f:
        # Iterate over the lines in the input file
        for line in input_f:
            # Get the token count of the line
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(line)
            line_token_count = len(tokens)
            
            # If the token count is >8000, split the line into as many equal-sized chunks as needed so each chunk is <8000 tokens
            if line_token_count > 8000:
                num_chunks = line_token_count // 8000
                print(f'{input_file} is too long ({line_token_count} tokens), splitting into {num_chunks+1} chunks')
                for i in range(num_chunks+1):
                    chunk = line[i*8000:(i+1)*8000]
                    # Get the embedding for the chunk
                    embedding = get_embedding(chunk, engine='text-embedding-ada-002')
                    # Write out the chunk embedding to its own file
                    print(f'Writing chunk {i+1} of {num_chunks+1} to {output_file}-chunk-{i+1}')
                    with open(output_file + "-chunk-" + str(i+1), 'w') as chunk_f:
                        chunk_f.write(str(embedding) + '\n')
            else:
                # Get the embedding for the line
                embedding = get_embedding(line, engine='text-embedding-ada-002')
                # Write the embedding to the output file as a string
                with open(output_file, 'w') as output_f:
                    output_f.write(str(embedding) + '\n')

def main():
    import argparse
    import glob
    import os

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the required input and output directory arguments
    parser.add_argument('input_dir', help='The input directory path')
    parser.add_argument('output_dir', help='The output directory path')

    # Add the optional --subdir or --file flag and pattern arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--subdir', help='A glob pattern for matching subdirectories')
    group.add_argument('--file', help='A glob pattern for matching files')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the input and output directory paths from the command-line arguments
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Initialize empty list for subdirectories or files to process
    subdirs_or_files = []

    # Check if the --subdir or --file flag was provided
    if args.subdir:
        # Use glob to match the subdirectory pattern
        subdirs_or_files = glob.glob(os.path.join(input_dir, args.subdir))
    elif args.file:
        # Iterate over all the subdirectories in the input directory
        for root, dirs, files in sorted(os.walk(input_dir)):
            # Use glob to match the file pattern for each subdirectory
            file_matches = glob.glob(os.path.join(root, args.file))
            # Add the matching files to the list of subdirectories or files to process
            subdirs_or_files.extend(file_matches)

    # Iterate over all the subdirectories or files in the input directory
    for item in sorted(subdirs_or_files):
        # Check if the item is a file or a subdirectory
        if os.path.isfile(item):
            # Process the file and write the output to an identically named file in the output directory
            input_file = item
            file = os.path.basename(item)
            output_file = os.path.join(output_dir, file)
            # Skip the file if the output file already exists and is more than 0 bytes
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f'Skipping {item} because output file already exists')
                continue
            # Skip the file if a chunk of the output file already exists
            if os.path.exists(output_file + "-chunk-1"):
                print(f'Skipping {item} because output file chunks already exist')
                continue
            # Print the file name
            print(f'Processing {item}')
            get_embeddings(input_file, output_file)
        else:
            # Create the same subdirectories in the output directory
            subdir = item.replace(input_dir, output_dir)
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            # Iterate over all the files in the subdirectory
            print(f'Processing {subdir}')
            for file in sorted(os.listdir(item)):
                input_file = os.path.join(item, file)
                output_file = os.path.join(subdir, file)
                # Skip the file if the output file already exists and is more than 0 bytes
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f'Skipping {subdir}/{file} because output file already exists')
                    continue
                # Skip the file if a chunk of the output file already exists
                if os.path.exists(output_file + "-chunk-1"):
                    print(f'Skipping {subdir}/{file} because output file chunks already exist')
                    continue
                # Print the file name
                print(f'Processing {subdir}/{file}')
                # Process the file and write the output to an identically named file in the output subdirectory
                get_embeddings(input_file, output_file)

if __name__ == '__main__':
    main()
