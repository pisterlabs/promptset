import os
import pandas as pd
# pandas == 1.4.4
import PyPDF2
# pypdf == 2.10.5
import openai
# pip install openai==0.28.0
import json

# Initialize OpenAI API with your API key
openai.api_key = "YOUR-API-KEY"

# Check if the synthetic_data_raw.csv file already exists in the directory
if os.path.exists("synthetic_data_raw.csv"):
    main_df = pd.read_csv("synthetic_data_raw.csv")
else:
    main_df = pd.DataFrame()

# Define the path to the folder containing the PDFs
pdf_folder = "path"

# List all PDF files in the folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Function for breaking a single PDF page into a list of paragraphs with GPT4
def gpt4_conversation(conversation_log):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation_log,
        temperature=0
    )
    conversation_log.append({
        'role': response.choices[0].message.role,
        'content': response.choices[0].message.content.strip()
    })
    paragraphs = conversation_log[-1]['content'].split('\n')
    paragraphs = [para for para in paragraphs if len(para) > 10]
    
    return paragraphs

# Function for iterating over a PDF document, page by page, extracting paragraphs and creating a CSV file
def parse_pdf_with_paragraphs(filename, page_num, prompt, pdf_name):
    pdf = open(filename, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf)
    data = []

    page = pdf_reader.pages[page_num - 1]
    page_text = page.extract_text()

    # Replace page breaks with spaces
    page_text = page_text.replace('\n', ' ')

    conversation_log = [{'role': 'user', 'content': prompt + page_text}]
    paragraphs = gpt4_conversation(conversation_log)

    for para_num, paragraph in enumerate(paragraphs):
        data.append({
            'page': page_num,
            'para_num': para_num + 1,
            'text': paragraph,
            'pdf_name': pdf_name
        })

    df = pd.DataFrame(data)
    return df

# Adding paragraphs from a single page of the document into the main datasrt
def append_to_main_df(df, main_df):
    main_df = main_df.append(df, ignore_index=True)
    return main_df

# Execution loop
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_name = pdf_file[:-4]

    # Load a set of processed pages for the current document
    if not main_df.empty and pdf_name in main_df["pdf_name"].unique():
        processed_pages = set(main_df[main_df["pdf_name"] == pdf_name]["page"])
    else:
        processed_pages = set()

    # Create the pdf_reader object for the current PDF file
    pdf = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf)

    # Replace these values with your actual prompt and heading_dict
    prompt = ''' You are given text from a page of a research paper. Please break down the text below into a list of paragraphs contained in the text, with each paragraph on a new line. This is an example of how two (or more) paragraph should be represented:

Traditionally, teaching in physics at the university level is dominated by lectures and lab exercises. However, lectures are limited in their effectiveness of conveying certain kinds of knowledge, since students are passive participants [1]. It has been shown that a learning environment in which students are active participants can more efficiently develop students’ competences and increase their information retention [1, 2]. To transform physics classes into an active learning environment, we can change the format of the lectures [3, 4], and we can offer the students new virtual learning environments and methods to enhance their studies [5].
Virtual learning environments (VLE) can have different design philosophies. For instance, the Institute of Physics New Quantum Curriculum [6] teaches quantum mechanics through a series of texts and simulations. It was built on established PhET Look and Feel design principles [7, 8], which encourage the use of an open and exploratory design for simulations. Other VLE’s also use simulations to teach quantum mechanics in a similar manner [9–13].

Please do not just try to attempt to separate them by page breaks or whitespace. Break them by semantics, where you think one paragraph ends and where one begins. There should be approximately 5 paragraphs per the input text, but there can be more or less. Please do not exclude any text, unless it seems like it is from a footer or a header. If you deem that the text on the page does not contain meaningful paragraphs, for example if the text represents table of contents or something similar, you should just return 'no meaningful paragraphs found'. 
Here is the text: \n\n
'''

    print(f"Processing document: {pdf_file}")

    # Process the PDF page by page and get a DataFrame
    for page_num in range(1, len(pdf_reader.pages) + 1):
        if page_num in processed_pages:
            print(f"Skipping page {page_num} of {pdf_file} (already processed)")
            continue

        page_data = parse_pdf_with_paragraphs(pdf_path, page_num, prompt, pdf_name)
        main_df = append_to_main_df(page_data, main_df)

        # Save the main_df to a CSV file after processing each page
        main_df.to_csv("synthetic_data_raw.csv", index=False, escapechar='\\')

        # Print which page has been processed
        print(f"Processed page {page_num} of {pdf_file}")

# Save the final main_df to a CSV file after processing the entire document
main_df.to_csv("synthetic_data_raw.csv", index=False, escapechar='\\')

print("Done!")