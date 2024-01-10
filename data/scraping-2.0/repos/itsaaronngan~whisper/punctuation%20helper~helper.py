from openai import OpenAI
import os
from pathlib import Path
import json
import re

client = OpenAI()


# Function to find transcription files in the specified directory
def find_transcription_files(directory):
    transcription_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):  # Assuming transcriptions are in .txt files
                transcription_files.append(Path(root) / file)
    return transcription_files

def remove_non_ascii(original_transcript):
    return ''.join(char for char in original_transcript if ord(char) < 128)

def split_text_into_chunks(text, max_tokens=4000):
    words = text.split(' ')
    chunks = []
    current_chunk = ''

    for word in words:
        if len((current_chunk + ' ' + word).strip()) <= max_tokens:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def punctuation_assistant(ascii_transcript):
    chunks = split_text_into_chunks(ascii_transcript)
    responses = []
    system_prompt = """ You are a helpful assistant that adds punctuation and paragraphs to text. Preserve the original words and only insert recommend paragraphs and only necessary punctuation such as periods, commas, capitalization, symbols like dollar signs or percentage signs, and formatting. Use only the context provided. If there is no context provided say, 'No context provided'\n"""

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        responses.append(response.choices[0].message.content)

    return ' '.join(responses)



def product_assistant(punctuation_edited_transcript):
    # Load the existing acronyms and their transcriptions
    try:
        with open('acronyms.json', 'r') as file:
            acronyms = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        acronyms = {}

    # Prepare the system prompt
    system_prompt = """
    You are an intelligent assistant specializing in Acronyms; your task is to process transcripts, ensuring that all acronyms and specialised terms are in the correct format. The full term should be spelled out followed by the acronym in parentheses. For example, '401k' should be transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)'. Names that are not acronyms (e.g. AusLeap) should not be transformed. Everyday words should not be transformed or changed, only acronyms. Create and append to the end a list of all transformations in the format [text] transformed into [transform] as well as any unknown acronyms. Unknown acronyms are things that are not listed on the common acronyms. Here are some common acronyms and their transcriptions: {}
    """.format(', '.join([f"'{acronym}': '{transcription}'" for acronym, transcription in acronyms.items()]))

    # Process the transcript
    chunks = split_text_into_chunks(punctuation_edited_transcript)
    responses = []
    new_acronyms = {}

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        response_content = response.choices[0].message.content
        responses.append(response_content)

        # Extract new acronyms from the response
        matches = re.findall(r'(\w+ \(\w+\))', response_content)
        for match in matches:
            full_form, acronym = match.split(' (')
            acronym = acronym[:-1]  # Remove the closing parenthesis
            if acronym not in acronyms:
                new_acronyms[acronym] = full_form

    # Save the new acronyms to a file
    with open('new_acronyms.json', 'w') as file:
        json.dump(new_acronyms, file)

    # Generate the transformations text
    transformations = "\n".join([f"'{acronym}' transformed into '{full_form} ({acronym})'" for acronym, full_form in new_acronyms.items()])

    return ' '.join(responses), transformations

def text_summary(product_edited_transcript):
    system_prompt = """
    You are an intelligent assistant specializing in summarizing meeting transcripts that are educational in nature. For the provided text, you will first produce a 5-10 word title for the text. Then you should produce a summary of the text that is no more than 3 sentences long. The summary should be a coherent paragraph that is grammatically correct and does not contain any spelling errors. Also generate a list of key learnings or key topics that were discussed in the meeting. Create a list of 3 suggestions about how you would use the learning and the content to create packaged content for public consumption. This could include templates (provide suggestions), learning resources, or other content that would be useful to the public. Use only the context provided, if no context is provided say no context. Your role is to
 analyze and adjust acronyms and specialised terminology in the text. Once you've done that, produce the summary and key learnings."""
    chunks = split_text_into_chunks(product_edited_transcript)
    responses = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        responses.append(response.choices[0].message.content)
    return " ".join(responses)
    

def main(directory):
    auto_continue = False
    
    # Get a list of transcription files
    transcription_files = list(find_transcription_files(directory))
    
    # Process each transcription file
    for i, file_path in enumerate(transcription_files):
        next_file = transcription_files[i + 1] if i + 1 < len(transcription_files) else None

        if not auto_continue:
            while True:
                if next_file:
                    print(f"The next file is: {next_file}")
                user_input = input("Do you want to continue to the next file? (y/n/s/g): ")
                if user_input.lower() == 'y':
                    break
                elif user_input.lower() == 'n':
                    return
                elif user_input.lower() == 's':
                    continue
                elif user_input.lower() == 'g':
                    auto_continue = True
                    print("auto_continue is now True")
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes, 'n' for no, 's' for skip, or 'g' for go ahead with all files.")

        print(f"Processing transcription file: {file_path}")

        # Read the original transcript
        with open(file_path, 'r') as file:
            original_transcript = file.read()
        print("Original transcript read.")

        # Process the transcript
        ascii_transcript = remove_non_ascii(original_transcript)
        print("Non-ASCII characters removed.")
        punctuation_edited_transcript = punctuation_assistant(ascii_transcript)
        print("Punctuation added.")
        product_edited_transcript, transformations = product_assistant(punctuation_edited_transcript)
        print("Product names edited.")

        # Generate the summary
        summary = text_summary(product_edited_transcript)
        print("Summary generated.")

        # Combine the edited transcript and the summary
        final_output = product_edited_transcript + "\n\n" + summary
        print("Final output prepared.")

   	 # Write the final output to a new file
        output_file_path = file_path.parent /"transcripts"/ ("EDITED_" + file_path.name)
        with open(output_file_path, 'w') as file:
            file.write(final_output)
        print(f"Saved edited transcript and summary to: {output_file_path}")

        # After processing the file, if auto_continue is False, ask for user input again
        if not auto_continue:
            while True:
                user_input = input("Do you want to continue to the next file? (y/n/s/g): ")
                if user_input.lower() == 'y':
                    break
                elif user_input.lower() == 'n':
                    return
                elif user_input.lower() == 's':
                    break
                elif user_input.lower() == 'g':
                    auto_continue = True
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes, 'n' for no, 's' for skip, or 'g' for go ahead with all files.")

if __name__ == "__main__":
    transcriptions_folder_path = "/Users/aaronnganm1/Documents/Coding/Whisper Transcription/output"  # Replace with the path to your output folder
    main(transcriptions_folder_path)