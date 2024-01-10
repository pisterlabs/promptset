import os
import openai
import fnmatch
import spacy


def find_files(path, extensions):
    found_files = []

    for root, _, filenames in os.walk(path):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, f'*.{ext}'):
                found_files.append(os.path.join(root, filename))

    return found_files

nlp = spacy.load("en_core_web_sm")

def split_code_into_chunks(code, max_tokens, overlap_tokens):
    tokens = code.split()
    chunks = []
    line_counts = []

    start_token = 0
    current_line = 1

    while start_token < len(tokens):
        end_token = start_token + max_tokens
        if end_token < len(tokens):
            while end_token > start_token and tokens[end_token - 1] != '\n':
                end_token -= 1
            if end_token == start_token:
                end_token = start_token + max_tokens

        chunk_tokens = tokens[start_token:end_token]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)

        next_line = current_line + chunk_text.count('\n')
        line_counts.append((current_line, next_line - 1))
        current_line = next_line

        start_token = end_token - overlap_tokens

    return chunks, line_counts


def scan_eosio_file(file_path, api_key, engine, max_tokens=1000, overlap_tokens=10):
    # Connect to the OpenAI API
    openai.api_key = api_key

    # Read the Solidity file content
    with open(file_path) as f:
        solidity_code = f.read()

    # Split the code into smaller chunks
    code_chunks, line_counts = split_code_into_chunks(solidity_code, max_tokens, overlap_tokens)

    vulnerabilities = []
    for i, (chunk, (line_start, line_end)) in enumerate(zip(code_chunks, line_counts)):
        total_chunks = len(code_chunks)
        if i == 0:
            prompt = (f"Analyze vulnerabilities in the following Solidity smart contract code. "
                      f"The code is divided into chunks. This is chunk {i+1}/{total_chunks} "
                      f"(lines {line_start}-{line_end}):\n{chunk}\n\n"
                      f"Please wait for all the chunks to complete before analyzing and "
                      f"providing your answer with the line numbers of where the vulnerabilities exist.")
        else:
            prompt = f"Here is chunk {i+1}/{total_chunks} (lines {line_start}-{line_end}):\n{chunk}\n\nPlease wait for all the chunks to complete before analyzing and providing your answer with the line numbers of where the vulnerabilities exist."


        #Make a request to the OpenAI API to analyze the Solidity file
        #result = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=1000, n=1, stop=None, temperature=0.5) #/v1/completions
        result = openai.ChatCompletion.create(model=engine, messages=[{"role": "system", "content": prompt}], max_tokens=1000, n=1, stop=None, temperature=0.5)
        #vulnerabilities.append(result.choices[0].text.strip()) #/v1/completions
        vulnerabilities.append(result['choices'][0]['message']['content'].strip())


    return "\n".join(vulnerabilities)



def main():
    api_key = os.environ['INPUT_OPENAI_API_KEY']
    engine = os.environ['INPUT_ENGINE']
    repo_path = os.getcwd()
    file_extensions = ['sol']
    target_files = find_files(repo_path, file_extensions)

    for file in target_files:
        print(f"Scanning {file}...")
        vulnerabilities = scan_eosio_file(file, api_key, engine)
        with open(f"{file}_vulnerabilities.txt", "w") as f:
            f.write("Service provided to you by www.sentnl.io.\n")
            f.write(f"Vulnerabilities found in {file}:\n{vulnerabilities}\n")

if __name__ == "__main__":
    main()
