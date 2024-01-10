import csv
import os
import magic
import html2text
import re
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import sys
import requests
import tempfile
import re
import glob
import json
#import shutil
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity


# Downloads the content of a URL to a file.
def download_url_to_file(url, file_path):
    """Downloads the content of a URL to a file.

    Parameters:
    url (str): The URL to download.
    file_path (str): The file path to save the content to.
    """
    # Download the content of the URL
    response = requests.get(url)
    content = response.content

    # Save the content to the file
    with open(file_path, 'wb') as fh:
        fh.write(content)

# Extracts the text from an HTML file and returns it as a string.
def extract_text_from_html(html_path):
    # Read the HTML file
    with open(html_path, "r") as html_file:
        html = html_file.read()
    
    # Extract the text from the HTML
    text = html2text.html2text(html)
    
    return text

# Extracts the text from a PDF file and returns it as a string.
def extract_text_from_pdf(pdf_path):
    """Extracts the text from a PDF file and returns it as a string.

    Parameters:
    pdf_path (str): The file path to the PDF file.

    Returns:
    str: The extracted text.
    """
    with open(pdf_path, 'rb') as fh:
        # Create a PDF resource manager object that stores shared resources
        rsrcmgr = PDFResourceManager()

        # Create a StringIO object to store the extracted text
        output = StringIO()

        # Create a TextConverter object to convert PDF pages to text
        device = TextConverter(rsrcmgr, output, laparams=LAParams())

        # Create a PDF page interpreter object
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page contained in the PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            interpreter.process_page(page)

        # Get the extracted text as a string and close the StringIO object
        text = output.getvalue()
        output.close()

        # Close the PDF file and text converter objects
        device.close()

    # Remove ^L page breaks from the text
    text = text.replace('\x0c', '\n')

    return text

# Splits text into a list of tuples, where each tuple contains a section header and the corresponding text.
def split_into_sections(text):
    """Splits a string of text into a list of tuples, where each tuple contains a section header and the corresponding text.

    Parameters:
    text (str): The input text to split into sections.

    Returns:
    list: A list of tuples, where each tuple contains a section header and the corresponding text.
    """
    # Use a regular expression to match the "References" section
    #pattern = r'(\n\nReferences[^\n]*)\n'
    # Also match the "References" section if it is preceded by # or ## (markdown-style headers)
    pattern = r'(\n\n(#+\s+)?References[^\n]*)\n'
    match = re.search(pattern, text)
    if match:
        # Remove the "References" section and everything that follows
        text = text[:match.start()]

    # If there is a section whose name starts with "Access", remove it
    pattern = r'(\n\nAccess[^\n]*)\n'
    #pattern = r'(\n\nAccess denied[^\n]*)\n'
    match = re.search(pattern, text)
    if match:
        # Remove the "Access denied" section and everything that follows
        text = text[:match.start()]

    # Use a regular expression to match and remove any base64 encoded data
    text = re.sub(r'data:.+;base64[^)]+', '', text)


    # Use a regular expression to split the text into sections
    #pattern = r'\n\n(\d+[\.:]\s+[^\n]+)\n\n'
    # Match section headers that start with a number followed by a period or colon,
    # or markdown-style headers that start with one to six hash marks followed by a space
    pattern = r'\n\n(#+\s+[^\n]+|\d+[\.:]\s+[^\n]+)\n\n'
    sections = re.split(pattern, text)
    print("Found", len(sections), "sections.")

    # Extract the section headers and their corresponding text
    headers = ["Title-Abstract"]
    content = []
    for i, section in enumerate(sections):
        if i % 2 == 0:
            # This is a section of content
            content.append(section)
        else:
            # This is a section header
            headers.append(section)
            #print(section)

    # Zip the section headers and content together
    sections = list(zip(headers, content))

    #print(headers)

    return sections

# Splits text into subsection tuples containing a section header and the corresponding text.
def split_section_into_subsections(section_header, section_content, enc, max_tokens=3000):
    """Splits a section of text into smaller parts, each of which is returned
    as a tuple containing a subsection header and the corresponding text.

    Parameters:
    section_header (str): The header for the section to be split.
    section_content (str): The content of the section to be split.
    enc (object): An encoder object used to encode the section content as a sequence of tokens.
    max_tokens (int, optional): The maximum number of tokens allowed in each subsection. Default is 3000.

    Returns:
    list: A list of tuples, where each tuple contains a subsection header and the corresponding text.
    """
    # Encode the section content as a sequence of tokens
    tokens = enc.encode(section_content)

    if len(tokens) <= max_tokens:
        # The section does not need to be split into subsections
        return [(section_header, section_content)]

    # Split the section into numbered subsections
    pattern = r'\n\n(\d+\.\d+[\.:]\s+[^\n]+)\n\n'
    subsections = re.split(pattern, section_content)

    # Extract the subsection headers and their corresponding text
    headers = [f"{section_header.split('.')[0]}. Section intro"]
    content = []
    for i, subsection in enumerate(subsections):
        if i % 2 == 0:
            # This is a subsection of content
            content.append(subsection)
        else:
            # This is a subsection header
            headers.append(subsection)

    # Zip the subsection headers and content together
    subsections = list(zip(headers, content))


    # Split any subsections that are still too long into smaller parts
    result = []
    for header, content in subsections:
        parts = split_subsection_into_paragraphs(header, content, enc, max_tokens)
        result.extend(parts)

    return result

# Splits a subsection by paragraphs if required based on a maximum number of tokens.
def split_subsection_into_paragraphs(subsection_header, subsection_content, enc, max_tokens=3000):
    """
    This function splits a subsection by paragraphs if required based on a maximum number of tokens.

    Parameters:
        subsection_header (str): The header of the subsection.
        subsection_content (str): The content of the subsection.
        enc (Encoder): An encoder object used to encode and decode the subsection content.
        max_tokens (int, optional): The maximum number of tokens allowed in each part. Defaults to 3000.

    Returns:
        list: A list of tuples containing the subsection header and content for each part.
    """
    # Encode the subsection content as a sequence of tokens
    tokens = enc.encode(subsection_content)

    if len(tokens) <= max_tokens:
        # The subsection does not need to be split into parts
        return [(subsection_header, subsection_content)]

    # Split the subsection into parts
    start = 0
    parts = []
    while start < len(tokens):
        # Calculate the size of the next part
        end = start + max_tokens

        # Find the nearest newline boundary within the part
        newline_pos = subsection_content[start:end].find('\n\n')
        if newline_pos != -1:
            end = start + newline_pos

        # Extract the part
        part_tokens = tokens[start:end]
        part_content = enc.decode(part_tokens)

        # Add the part to the list of parts
        parts.append((subsection_header, part_content))

        # Update the start index
        start = end + 2

    return parts

# Combines <1000 token subsections of text into larger chunks with a maximum of 2000 tokens.
def combine_subsections(subsections, enc):
    """
    This function combines subsections of text into larger chunks of text.
    It takes in a list of subsections, each containing a header and content, and an encoder object.
    It returns a list of combined subsections, each containing a header and content.
    The combined subsections are created by combining subsections that have less than 1000 tokens each,
    and the total number of tokens in the combined subsection is less than 2000.
    """
    # Initialize the list of combined subsections
    combined_subsections = []

    # Initialize the current combined subsection
    current_subsection_header = ""
    current_subsection_content = ""
    current_subsection_tokens = 0

    # Iterate through the subsections
    for header, content in subsections:
        # Encode the content as a sequence of tokens
        tokens = enc.encode(content)

        # If the current combined subsection has less than 1000 tokens and the current subsection has less than 1000 tokens, combine them
        if current_subsection_tokens + len(tokens) < 2000 and len(tokens) < 1000:
            # Update the current combined subsection header
            if current_subsection_header == "":
                current_subsection_header = header
                current_subsection_content = header + "\n"
            else:
                if current_subsection_header != header:
                    current_subsection_content += "\n\n" + header + "\n"
                #current_subsection_header += header

            # Update the current combined subsection content
            current_subsection_content += content

            # Update the current combined subsection token count
            current_subsection_tokens += len(tokens)
        else:
            # Add the current combined subsection to the list of combined subsections
            combined_subsections.append((current_subsection_header, current_subsection_content))

            # Reset the current combined subsection
            current_subsection_header = header
            current_subsection_content = header + "\n" + content
            current_subsection_tokens = len(tokens)

    # Add the final combined subsection to the list of combined subsections
    combined_subsections.append((current_subsection_header, current_subsection_content))

    return combined_subsections

# Splits a text into sections and writes each section to a separate text file.
def sectionize_content(text, content_file, enc):
    """
    This function takes in a text, content file, and encoding as parameters and splits the text into sections.
    It then writes each section to a separate text file, combining adjacent tuples with less than 1000 tokens
    until they exceed 1000 tokens.
    It also updates the subheader if there are multiple sequential identical subheaders.
    Finally, it writes the content to the output file if the subsection length is greater than 350 characters.
    """
    text = text
    content_file = content_file
    enc = enc

    # Split the text into sections
    sections = split_into_sections(text)

    # Write each section to a separate text file
    for header, content in sections:
        print("Header: ", header)
        # Split the section into subsections if necessary
        subsections = split_section_into_subsections(header, content, enc)

        # Combine adjacent tuples with less than 1000 tokens until they exceed 1000 tokens
        combined_subsections = combine_subsections(subsections, enc)

        # Initialize the counter for numbering sequential identical subheaders
        subheader_count = 1

        # Process each combined subsection
        for subheader, subcontent in combined_subsections:
            # Update the subheader if there are multiple sequential identical subheaders
            if subheader_count > 1:
                subheader += f"-part{subheader_count}"
            subheader_count += 1

            # Use tiktoken to encode the subsection content as a sequence of tokens
            subcontent_tokens = enc.encode(subcontent)

            # Get the name of the output file
            # print("Subheader: ",subheader)
            section_name = re.sub(r'[^a-zA-Z0-9]', '', subheader.replace('/', '-'))

            # print("Section name: ",section_name)
            output_path = str(content_file) + "." + section_name + ".section.txt"

            #if len(subcontent) == 0:
            # If the subsection length is <350 characters, skip it
            if len(subcontent) < 350:
                subheader_count = subheader_count - 1
            else:
                # Write the content to the output file
                with open(output_path, 'w') as f:
                    f.write(subcontent)
                print(
                    f"{subheader} ({len(subcontent)} characters, {len(subcontent_tokens)} tokens) written to {output_path}")

# Download URL content, extract text, and write each section to a separate text file.
def download_and_extract(csv_filename):
    """
    This function uses a CSV file containing URLs and extracts the text from each URL.
    It then sectionizes the content and writes each section to a separate text file.

    Inputs:
    csv_filename: The name of the CSV file containing the URLs

    Outputs:
    downloaded_files: An array of all the downloaded file names
    """
    # Create a tmp/ directory to store the downloaded files in the same directory as the csv file
    csv_dir = os.path.dirname(csv_filename)
    temp_dir = os.path.join(csv_dir, 'tmp')
    # Make sure the temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)
    print (f"Using {temp_dir} as the temporary directory")
    
    # Use /tmp/systematic_reviews as the temporary directory for now
    #temp_dir = '/tmp/systematic_reviews'
    # Make sure the temporary directory exists
    #os.makedirs(temp_dir, exist_ok=True)

    # Create an array of all the downloaded file names
    downloaded_files = []

    # Open the CSV file
    with open(csv_filename, 'r') as csv_file:
        print(f"Reading CSV file: {csv_filename}")
        reader = csv.DictReader(csv_file)

        # Create a new CSV file to store the extracted sections
        #output_filename = os.path.splitext(csv_filename)[0] + '_sections.csv'
        #with open(output_filename, 'w') as output_file:
            #writer = csv.DictWriter(output_file, fieldnames=['ArticleURL', 'SectionHeader', 'SectionText'])
            #writer.writeheader()

            # Iterate over each row in the CSV file
        for row in reader:
            # Get the URL from the row
            url = row['ArticleURL']
            # If URL is blank, skip this row
            if url == '':
                continue
            print(f"Processing URL: {url}")
            


            # Download the URL to a file in the temporary directory
            print(f"Downloading URL to file: {url}")
            file_name = os.path.basename(url)
            if file_name == '':
                file_name = url.rsplit('/', 2)[-2]
            # If the file name is longer than 100 characters, use the part after the last non-alphanumeric, non-period and non-underscore character
            #if len(file_name) > 100:
                # Split on any non-alphanumeric, non-period and non-underscore special character
                #file_name = re.split(r'[^a-zA-Z0-9_.]', file_name)[-1]
            # If it's still longer than 100 characters, truncate it
            if len(file_name) > 100:
                file_name = file_name[:100]

            # Remove non-alphanumeric, non-period and non-underscore characters from the file name
            #file_name = re.sub(r'[^\w_]', '', file_name)  
            file_name = re.sub(r'[^\w_.]', '', file_name)
            print(f"File name: {file_name}")
            file_path = os.path.join(temp_dir, file_name)
            # If the file already exists, skip it
            if os.path.exists(file_path):
                print(f"File already exists: {file_path}")
            else:
                download_url_to_file(url, file_path)
            downloaded_files.append(file_path)

            # Identify the file type
            file_type = magic.from_file(file_path)

            # Extract the text from the file
            if file_type.startswith('HTML document'):
                text = extract_text_from_html(file_path)
            elif file_type.startswith('PDF document'):
                text = extract_text_from_pdf(file_path)
            else:
                raise ValueError('Unsupported file type: {}'.format(file_type))

            # Use a regular expression to match and remove any base64 encoded data
            text = re.sub(r'data:.+;base64[^)]+', '', text)

            # Write the full text to a .txt file
            full_text_filename = os.path.splitext(file_name)[0] + '.full.txt'
            full_text_file_path = os.path.join(temp_dir, full_text_filename)
            # If the file already exists, skip it
            if os.path.exists(full_text_file_path):
                print(f"Full text file already exists: {full_text_file_path}")
            else:
                print(f"Writing full text to file: {full_text_file_path}")
                with open(full_text_file_path, 'w') as fh:
                    fh.write(text)

            
            # encode the text as a sequence of tokens
            enc = tiktoken.get_encoding("gpt2")
            #enc = GPT2TokenizerFast.from_pretrained("gpt2")

            tokens = enc.encode(text)

            print(f"Total token count: {len(tokens)}")

            # Sectionize content
            sectionize_content(text, file_path, enc)

        # Return the list of downloaded files
        return downloaded_files

# Get embeddings from an input file and write them to an output file
def get_embeddings(input_file, output_file):
    """
    This function takes an input file and an output file as parameters.
    It reads the input file line by line and calculates the token count of each line.
    If the token count is less than 8000, it calculates the embedding using the
    text-embedding-ada-002 engine and writes the embedding to the output file as a string.
    """
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
                # Error out for now
                raise ValueError(f'{input_file} is too long ({line_token_count} tokens)')
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

# Combine filename, text, and embeddings to a JSON file
def combine_text_and_embeddings_to_json(downloaded_files):
    """
    This function combines text and embeddings from a list of downloaded files and writes them to a JSON file.
    The json file contains a list of dictionaries, each dictionary containing the file name, text, and embedding.

    Parameters:
    downloaded_files (list): A list of downloaded files.

    Returns:
    None. Writes the combined file name, text, and embeddings to a JSON file.
    """
    # Create a list to store the combined text and embeddings
    combined_text_and_embeddings = []

    # For each downloaded file
    for file in downloaded_files:
        # Find all $file.*.section.txt files
        section_files = glob.glob(file + '.*.section.txt')
        # For each section file, get the embedding
        for section_file in section_files:
            # Get the embedding file name
            embedding_file = section_file + '.embedding'
            # Read the embedding file
            with open(embedding_file, 'r') as f:
                embedding = f.read()
            # The embedding is a string, so convert it to a list of floats, split on commas, and remove the brackets
            embedding = [float(x) for x in embedding[1:-2].split(',')]

            # Read the section file
            with open(section_file, 'r') as f:
                text = f.read()
            # Extract just the file name from section_file
            section_file_basename = os.path.basename(section_file)
            # Create a dictionary with the text, embedding, and file name
            text_and_embedding = {
                'file_name': section_file_basename,
                'text': text,
                'embedding': embedding
            }
            # Add the dictionary to the list
            combined_text_and_embeddings.append(text_and_embedding)

    # Create the output file name in the same directory as the downloaded files
    temp_dir = os.path.dirname(downloaded_files[0])
    output_file = os.path.join(temp_dir, 'combined_text_and_embeddings.json')

    #output_file = 'combined_text_and_embeddings.json'
    # Write the list to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_text_and_embeddings, f, indent=4)
    print(f"Wrote combined text and embeddings to {output_file}")         

    return output_file   


def update_csv_with_section_counts(csv_filename):
    """
    This function updates a CSV file with a SectionsFound header, counts the number of .embedding files present on disk for each URL, and populates the value into the SectionsFound column for each row. It then writes out the modified file to a .out.csv file with the same basename (without extension) as the original csv_filename.

    Args:
        csv_filename (str): The name of the CSV file containing the articles to be downloaded and extracted.

    Returns:
        output_filename (str): The name of the output CSV file.
    """
    # Get the base filename (without extension) from the input filename
    base_filename = os.path.splitext(csv_filename)[0]
    # Create the output filename
    output_filename = base_filename + '.indexed.csv'

    # Read the input CSV file
    with open(csv_filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        # Create a new CSV file to store the updated info
        with open(output_filename, 'w') as output_file:
            # Add the SectionsFound header to the output CSV file
            fieldnames = reader.fieldnames + ['SectionsFound']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            # Iterate over each row in the CSV file
            for row in reader:
                # Get the URL from the row
                url = row['ArticleURL']
                # If URL is blank, skip this row
                if url == '':
                    continue

                # Find all $file.*.section.txt.embedding files
                # Create the file path
                file_name = os.path.basename(url)
                if file_name == '':
                    file_name = url.rsplit('/', 2)[-2]
                # Truncate file_name if it's longer than 100 characters
                if len(file_name) > 100:
                    file_name = file_name[:100]
                # Remove non-alphanumeric, non-period and non-underscore characters from the file name
                file_name = re.sub(r'[^\w_.]', '', file_name)
                print(f"file_name: {file_name}")
                csv_dir = os.path.dirname(csv_filename)
                temp_dir = os.path.join(csv_dir, 'tmp')
                file_path = os.path.join(temp_dir, file_name)
                print(f"Looking for section files in {file_path}")
                section_files = glob.glob(file_path + '.*.section.txt.embedding')
                print(f"Found {len(section_files)} section files for {url}")
                # Count the number of section files
                section_count = len(section_files)

                # Update the row with the new info
                row.update({'SectionsFound': section_count})
                # Write the row to the output file
                writer.writerow(row)

    # Return the name of the output file
    return output_filename

# Download and extract articles from a CSV file and get embeddings
def main():
    """
    This function downloads and extracts articles from a given CSV file, gets the embeddings for each section text file, and combines all the section text files and their corresponding embeddings into a single JSON file.

    Args:
        csv_filename (str): The name of the CSV file containing the articles to be downloaded and extracted.

    Returns:
        None
    """
    # Check that a filename was passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python download_articles_and_embeddings.py <filename>.csv")
        sys.exit(1)

    # Get the filename from the command line arguments
    csv_filename = sys.argv[1]

    # Check that the file exists
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found")
        sys.exit(1)

    downloaded_files = download_and_extract(csv_filename)
    print(f"Downloaded {len(downloaded_files)} files")
    #print(downloaded_files)

    # For each downloaded file, get the embedding of all the section text files
    for file in downloaded_files:
        # Find all $file.*.section.txt files
        section_files = glob.glob(file + '.*.section.txt')
        # For each section file, get the embedding
        for section_file in section_files:
            # Get the output file name
            output_file = section_file + '.embedding'
            # if the output file already exists, skip it
            if os.path.exists(output_file):
                print(f"Output file already exists: {output_file}")
                continue
            # Get the embedding
            print(f"Getting embedding for {section_file} and writing to {output_file}")
            get_embeddings(section_file, output_file)
        
    # Combine all the section text files and their corresponding embeddings into a single json file
    json_file = combine_text_and_embeddings_to_json(downloaded_files)

    # Update the CSV file with the number of sections found for each article
    output_filename = update_csv_with_section_counts(csv_filename)
    print(f"Updated CSV file with section counts: {output_filename}")

    # Find the path (not including filename) to the currently running script
    script_path = os.path.dirname(os.path.realpath(__file__))


    print("To run the next step, run:\n" + \
        f"python {script_path}/answer_questions.py {output_filename} {json_file} <questions.csv or string>")


if __name__ == '__main__':
    main()
