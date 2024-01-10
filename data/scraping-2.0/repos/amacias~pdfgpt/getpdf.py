# Based on the work from: 
# https://anirudhlohia.medium.com/automated-summarisation-of-pdfs-with-gpt-and-python-8cb398e5f029

#export OPENAI_API_KEY=key_copied_from_openai_site

from PyPDF2 import PdfReader
import sys

# This function is reading PDF from the start page to final page
# given as input (if less pages exist, then it reads till this last page)
def get_pdf_text(document_path, start_page=1, final_page=999):
    reader = PdfReader(document_path)
    number_of_pages = len(reader.pages)
    print('Document contains',  number_of_pages, ' pages.')
    page = []

    for page_num in range(start_page - 1, min(number_of_pages, final_page)):
        page += reader.pages[page_num].extract_text()

    #original_stdout = sys.stdout # Save a reference to the original standard output

    #with open('parsed.txt', 'w') as f:
    #    sys.stdout = f # Change the standard output to the file we created.
    #    print(page)
    #    sys.stdout = original_stdout # Reset the standard output to its original value

    return page

import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
def gpt_req_res(subject_text='write an essay on any subject.',
                prompt_base='answer like an experienced consultant: ',
                model='text-davinci-003',
                max_tokens=1200,
                temperature=0.8):

    # https://platform.openai.com/docs/api-reference/completions/create
    response = openai.Completion.create(
        model='gpt-3.5-turbo',
        #prompt=prompt_base + ': ' + subject_text,
        prompt=(prompt_base, subject_text),
        temperature=temperature,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text

doc_path_name = 'test.pdf'
doc_text = get_pdf_text(doc_path_name, 1, 2)

prompt = 'summarize like an experienced consultant in 5 bullets: '
reply = gpt_req_res(doc_text, prompt)
print(reply)