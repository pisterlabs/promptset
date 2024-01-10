#### progress.py

# Copyright (c) 2023 Landon Dahle
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# For more information on licensing, see https://opensource.org/licenses/MIT

# -----------------------------------------------------------------------------
# Author: Landon Dahle
# Date: 2023
# Project: Progress - BMEN 351 Final Project
# License: MIT
# -----------------------------------------------------------------------------

# =============================================================================
# Input Variables
# =============================================================================


#### Modules
# General
import time
import os
import sys

# Text Extraction
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import csv
import re

# CSV Parsing
import pandas as pd
import logging
import multiprocessing
from multiprocessing import Pool

# OpenAI API
from openai import OpenAI

#### Frontend Inputs
# User Search Queary
# API Token
#### OpenAI API Calls
api_key = "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"
 
# Instantiate client (API key is read from environment by default)
client = OpenAI(api_key=api_key)
 
# Define the Assistant ID
assistant_id = "asst_Rsv4QUDEo7l7ukfmICBOkeZF"
 
# Create a Thread for the conversation
thread = client.beta.threads.create()


#### Backend Inputs
# data storage path
# 

# =============================================================================
# OpenAI_API
# =============================================================================

def process_text(text):
    # Add the user's text to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text
    )

    # Run the Assistant on the Thread using the specified Assistant ID
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id  # Use the specific Assistant ID
    )

    # Check the run status and retrieve results
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    # Retrieve and display the Assistant's response
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    for message in messages:
        if message.role == 'assistant':
            return message.content[0].text.value



# =============================================================================
# Paper_Databasing
# =============================================================================

def clean_text(text):
    # Replace or remove invalid characters and character sequences
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Remove new lines
    cleaned_text = re.sub(r'[\r\n]+', ' ', cleaned_text)
    
    # # All above in one query
    # cleaned_text = re.sub(r'[^\x00-\x7F]+|\s+|[\r\n]+', ' ', text)
    
    return cleaned_text

def pdf_to_csv(input_pdf_path, output_csv_path):
    # Extract Text
    try:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)  # Remove 'codec' argument
        
        with open(input_pdf_path, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()
            
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, 
                                          password=password, caching=caching, 
                                          check_extractable=True):
                interpreter.process_page(page)
            
            text = retstr.getvalue()
        
        device.close()
        retstr.close()
        
        cleaned_text = clean_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=\.|!|\?)(?<!\d\.\d)(?!\s*[a-z]) *', cleaned_text)
        
        # Merge small sentences
        min_char_count = 40  # Minimum character count for a sentence to be considered complete
        min_space_count = 5  # Minimum space count for a sentence to be considered complete
        
        # Function to check if a sentence is below the thresholds
        def is_sentence_short(sentence):
            return len(sentence) < min_char_count or sentence.count(' ') < min_space_count
        
        # Iterate through sentences and append short sentences to the previous one
        i = 0
        while i < len(sentences):
            if i > 0 and is_sentence_short(sentences[i]):
                sentences[i-1] += ' ' + sentences.pop(i)
            else:
                i += 1

        # Function to check if a sentence has more than one semicolon
        def has_multiple_semicolons(sentence):
            return sentence.count(';') > 1
        
        # Iterate through sentences and remove those with multiple semicolons
        sentences = [sentence for sentence in sentences if not has_multiple_semicolons(sentence)]

        # # Create rolling window text chunks
        # rolling_window_size = 3
        # text_chunks = []
        # for i in range(len(sentences) - rolling_window_size + 1):
        #     chunk = ' '.join(sentences[i:i + rolling_window_size])
        #     text_chunks.append(chunk)

        # Write the text chunks to the CSV file
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Text"])  # Header row
            # csv_writer.writerows([[chunk] for chunk in text_chunks])
            csv_writer.writerows([sentence] for sentence in sentences)
    except Exception as e:
        logging.error(f"Error occurred while processing PDF to CSV with rolling window: {e}")
                    
def process_pdf(args):
    """
    Process a single PDF and save its text content to a corresponding CSV file.
    
    :param args: Tuple containing input PDF file path and output CSV file path.
    """
    try:
        pdf_file_path, output_csv_path = args
        pdf_to_csv(pdf_file_path, output_csv_path)
    except Exception as e:
        logging.error(f"Error occurred while processing PDF: {pdf_file_path}. Error: {e}")

def process_pdfs(PDF_input_directory, CSV_output_directory):
    """
    Process all PDFs in the input directory and save their text content to corresponding CSV files in the output directory.
    
    :param PDF_input_directory: Path to the directory containing input PDFs.
    :param CSV_output_directory: Path to the directory where output CSVs will be saved.
    """
    try:
        # Create a list of tuples, each containing an input PDF file path and the corresponding output CSV file path
        tasks = []
        for pdf_file in os.listdir(PDF_input_directory):
            if pdf_file.lower().endswith('.pdf'):
                pdf_file_path = os.path.join(PDF_input_directory, pdf_file)
                output_csv_path = os.path.join(CSV_output_directory, os.path.splitext(pdf_file)[0] + '.csv')
                
                # Check if the CSV file already exists
                if not os.path.exists(output_csv_path):
                    tasks.append((pdf_file_path, output_csv_path))

        with multiprocessing.Pool() as pool:
            # Process each PDF in parallel
            pool.map(process_pdf, tasks)

    except Exception as e:
        logging.error(f"Error occurred while processing PDFs: {e}")

# =============================================================================
# Keyword_Search
# =============================================================================

# Define your keywords list here
keywords = [
    # Original keywords
    'further research', 'future study', 'future work', 'suggest that', 'recommend', 
    'should be investigated', 'should be explored', 'additional studies', 'subsequent research', 
    'more research', 'requires further', 'requires additional', 'warrants further', 
    'investigation needed', 'explore in depth', 'unanswered questions', 'open questions', 
    'remaining questions', 'directions for future', 'scope for future', 'prospects for future', 
    'it would be interesting to', 'call for more', 'imperative to study', 'need for more research', 
    'need for further study', 'further investigation', 'further analysis', 'extended research', 
    'continued research', 'ongoing study', 'potential for further', 'potential for additional', 
    'points for future research', 'leads to further', 'raises the question of', 
    'to be addressed in future', 'to be addressed in subsequent', 'room for further', 
    'gap in the literature', 'literature gap', 'expanding upon', 'build upon the findings', 
    'beyond the scope of this', 'leave open the possibility', 'further studies will', 
    'warrant additional', 'further empirical study', 'further theoretical work', 
    'further experimental work', 'future lines of inquiry', 'avenues for future research', 
    'avenues for further investigation', 'extend this line of research', 'extend these findings', 
    'deepen our understanding', 'broaden our understanding', 'further elucidate', 'further define', 
    'further clarify', 'further examine', 'further explore', 'advancing the field', 
    'stimulate further research', 'prompts further investigation', 'fuel further study', 
    'lay the groundwork for future', 'set the stage for future', 'a foundation for future', 
    'to pave the way for future', 'serve as a basis for future', 'encourage further exploration', 
    'highlight the need for', 'underscore the importance of', 'catalyze future research', 
    'spur further investigation', 'propel the field forward', 'next steps include', 
    'follow-up studies', 'further insights', 'elaborate on', 'expand upon', 'build on these results', 
    'build on this work', 'subsequent work could', 'additional work is needed to', 
    'further work is required to', 'merits further attention', 'prioritize for future study', 
    'imperative for future research', 'critical for future studies', 'integral to future work',

    # Expanded and modified keywords
    'call for further exploration', 'call for detailed study', 'advocate for further research', 
    'highlight the potential for', 'indicate the need for', 'suggest future directions', 
    'warrant a deeper look', 'invite further inquiry', 'open avenues for research', 
    'point towards future work', 'suggest additional studies', 'require more detailed investigation',
    'demand further examination', 'entail further scrutiny', 'entail additional investigation',
    'prompt further analysis', 'invite more research', 'encourage detailed studies',
    'implications for future research', 'prospects for in-depth study', 'prospects for detailed investigation',
    'recommend further inquiry', 'recommend additional investigation', 'suggests avenues for exploration',
    'entails future study', 'entails further exploration', 'opens the door for future research', 
    'invites further studies', 'necessitates further research', 'necessitates additional studies', 
    'future research should include', 'additional research should explore', 'further study is warranted',
    'further investigation is warranted', 'suggests paths for future research', 'indicates areas for further study',
    'points to the need for', 'further work is needed on', 'additional work is warranted on', 
    'further analysis is required on', 'further exploration is needed in', 'further studies are needed in',
    'further investigation is needed in', 'additional studies are required in', 'expand the scope of investigation',
    'expand the scope of research', 'expand the scope of study', 'future research could explore', 
    'future studies could investigate', 'future work could examine', 'necessitates a broader examination',
    'necessitates a broader investigation', 'suggests a need for broader study', 'suggests a need for in-depth research',
    'suggests a need for comprehensive study', 'suggests potential areas for research', 
    'suggests potential areas for exploration', 'opens new avenues for research', 
    'opens new avenues for investigation', 'future research might consider', 'future studies might consider', 
    'further research might explore', 'further research might investigate', 'raises potential areas for study',
    'raises potential areas for research', 'indicates a direction for future work', 
    'indicates a direction for further study', 'indicates a direction for further research',
    
    # Expanded from false negatives
    'It is not yet clear', '?'
]



def classify_text(text):
    """
    Classify text for requests or advice for future research based on keywords.
    
    :param text: The text to classify.
    :return: 1 if the text contains requests or advice for future research, otherwise 0.
    """
    if any(keyword in str(text).lower() for keyword in keywords):
        return 1
    else:
        return 0

def classify_text_withAPI(text, existing_research_advice=None,  word_limit=300):
    """
    Classify text for requests or advice for future research based on keywords.
    If classified as 1 or existing research advice is 1, call OpenAI assistant for further processing.
    
    :param text: The text to classify.
    :param existing_research_advice: The existing classification, if any.
    :return: A tuple containing the classification (0 or 1) and the OpenAI assistant's output.
    """
    # Convert text to string and check the word count
    text_str = str(text)
    word_count = len(text_str.split())

    # Initialize default output for OpenAI
    openai_output = "0"
    
    # If the word count exceeds the limit, return without processing
    if word_count > word_limit:
        return (0, "0")

    # Check if existing_research_advice is 1, if so, directly make the OpenAI call
    if existing_research_advice == 1:
        openai_output = process_text(text)
        return (1, openai_output)
    elif existing_research_advice == 0:
        return (0, "")
    elif any(keyword in str(text).lower() for keyword in keywords):
        openai_output = process_text(text)
        return (1, openai_output)
    else:
        return (0, "")

def process_row(row, columns_present):
    # Initialize variables
    research_advice = None
    openai_bool = ''
    openai_subjects = ''
    openai_output = ''

    try:
        
        research_advice = row.get('Research_Advice', '')

        # If research advice is already present and is positive
        if pd.notna(research_advice) and research_advice == 1:
            openai_bool = row.get('OpenAI_Bool', '')  # Use .get with default value
            openai_subjects = row.get('OpenAI_Subjects', '')  # Use .get with default value
            if openai_bool == '':
                # Call OpenAI for rows that have positive research advice but no OpenAI output
                research_advice, openai_output = classify_text_withAPI(row['Text'], existing_research_advice=1)
        else:
            # If research advice is not present, classify and process the text
            research_advice, openai_output = classify_text_withAPI(row['Text'])

    
        # Process OpenAI output if it exists
        if openai_bool == '' and research_advice == 1:
            try:
                if openai_output.strip() == '0':
                    openai_bool = '0'
                    openai_subjects = ''
                    
                else:
                    openai_parts = openai_output.split(',', 1)
                    if len(openai_parts) == 2 and openai_parts[0].strip() in ['0', '1']:
                        openai_bool, openai_subjects = openai_parts
                    else:
                        # Re-run classify_text_withAPI with enhanced prompt if openai_bool is not valid
                        enhanced_text = "Classify the following extract only as described: \"{}\"".format(row['Text'])
                        _, openai_output = classify_text_withAPI(enhanced_text)
                        openai_parts = openai_output.split(',', 1)
                        if openai_output and len(openai_parts) == 2:
                            openai_bool, openai_subjects = openai_parts
                        else:
                            openai_bool, openai_subjects = 'Error', 'Error'
                            
            except Exception as e:
                logging.error(f"Error processing OpenAI output for row: {row}. Error: {e}")
                # Set default values in case of an error
                return pd.Series((research_advice, 'Error', 'Error'))
    
        return pd.Series((row.get('Research_Advice', research_advice), openai_bool.strip(), openai_subjects.strip()))
    
    except Exception as e:
        logging.error(f"Error processing row: {row}. Error: {e}")
        # Set default values in case of an error
        return pd.Series((research_advice, 'Error', 'Error'))

def process_csv(args):
    """
    Process a single CSV and classify its text content, then save to a new CSV file.
    
    :param args: Tuple containing input CSV file path and output CSV file path.
    """
    try:
        csv_file_path, output_csv_path = args
        data = pd.read_csv(csv_file_path)
        
        # Drop the 'OpenAI_Output' column if it exists
        if 'OpenAI_Output' in data.columns:
            data.drop('OpenAI_Output', axis=1, inplace=True)
            
        if 'OpenAI_Bool' in data.columns:
            data.drop('OpenAI_Bool', axis=1, inplace=True)
        
        if 'OpenAI_Subjects' in data.columns:
            data.drop('OpenAI_Subjects', axis=1, inplace=True)
        
        if 'Temporal_Position' not in data.columns:
            # Calculate the temporal position of each text snippet
            data['Temporal_Position'] = data.index / (len(data) - 1)

        
        # Check for the presence of specific columns
        columns_present = data.columns
        
        # Apply process_row and split results into respective columns
        data[['Research_Advice', 'OpenAI_Bool', 'OpenAI_Subjects']] = data.apply(lambda row: process_row(row, columns_present), axis=1)
        
        # Save the modified DataFrame to CSV
        data.to_csv(output_csv_path, index=False)
        
    except Exception as e:
        logging.error(f"Error occurred while processing CSV: {csv_file_path}. Error: {e}")

def update_progress(result):
    global counter
    counter += 1
    print(f"Processed {counter} out of {total_files} files.")

def process_csvs(CSV_input_directory, CSV_output_directory):
    """
    Process all CSVs in the input directory and classify text content, 
    then save to corresponding new CSV files in the output directory.
    
    :param CSV_input_directory: Path to the directory containing input CSVs.
    :param CSV_output_directory: Path to the directory where output CSVs will be saved.
    """
    try:
        tasks = []
        for csv_file in os.listdir(CSV_input_directory):
            if csv_file.lower().endswith('.csv'):
                csv_file_path = os.path.join(CSV_input_directory, csv_file)
                output_csv_path = os.path.join(CSV_output_directory, os.path.splitext(csv_file)[0] + '.csv')
                tasks.append((csv_file_path, output_csv_path))

        global counter, total_files
        counter = 0
        total_files = len(tasks)

        with Pool() as pool:
            # Use map_async with a callback to update the progress
            result = pool.map_async(process_csv, tasks, callback=update_progress)
            result.wait()  # Wait for all tasks to complete

    except Exception as e:
        logging.error(f"Error occurred while processing CSVs: {e}")



# =============================================================================
# Outputs
# =============================================================================

if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python progress.py input_directory")
        sys.exit(1)

    data_directory = sys.argv[1] + r"\Data"
    
    
    #### Paper_Databasing
    # Define the input directory where PDFs are stored
    PDF_input_directory = os.path.join(data_directory, "PDFs")

    # Define the output directory where CSVs will be saved
    CSV_output_directory = os.path.join(data_directory, "CSVs")
    
    process_pdfs(PDF_input_directory, CSV_output_directory)
    
    print("Finished processing PDFs, processing CSV's")

    #### Keyword_Search 
    CSV_input_directory = CSV_output_directory # Vestigial, rewrites unless changed
    process_csvs(CSV_input_directory, CSV_output_directory)

    
    # Time
    end = time.time()
    print("The total runtime of the above code was",(end-start), "seconds")



# Backend:
    # Output CSV dataframe:
        # DOI, Title, Author, Date, file_path, Directions
            # file_path to pickle of pd DF that includes:
                # Seperated Text, Temporal Value, Processed Tokens, Classification

    
# Frontend:
    # Frequency Map of Directions
        # Graphical
        # CSV with phrase, frequency, and list of source articles
    # Output CSV dataframe:
        # DOI, Title, Author, Date, Directions