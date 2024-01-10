import os
import time
import openai
from PyPDF2 import PdfReader

# note that you need to apply for an OpenAI API key to try this demo
openai.api_key = 'YOUR-KEY'

def read_pdf(pdf_pathfile, maxlen = None):
    reader = PdfReader(pdf_pathfile)
    text = ""
    for page in reader.pages:
        text +=page.extract_text()
    if maxlen:
        if len(text) > maxlen:
            text = text[:maxlen]
    return text

def extract_info_from_pdf_chatgpt(url,savepath):
    filename = url_to_filename(url, save_path, replace_signs=['https://www.','\\','/','+','*','?','=','%','#'])
    if not os.path.exists(filename):
        return np.nan
    text = read_pdf(filename)
    gpt_abstract = request_chatgpt(text)
    return gpt_abstract

def extract_info_chatgpt(text, questions, prev_dict):
    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The previous info dict is %s" %prev_dict},
            {"role": "assistant", "content": "I have full understanding of this chunk of article: %s" %text},
            {"role": "user", "content": 'Extract these relevant information in English and organise it as "Python dict" (IMPORTANT), value set as "n/a" if not applicable and append if multiple answers come out. Be particularly cautious on the n/a in the previous info dict and try your best to update precise info. Questions: %s' %questions }
        ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=message,temperature=0)
    extracted_info = response['choices'][0]['message']['content']
    return response, extracted_info

def extract_info_from_text(text, questions, thegap=6000):
    steps = []
    prev_dict = {}
    responses = []
    for i in range(int(len(text)/thegap) + 1):
        paper_chunk = text[i*thegap: (i+1)*thegap]
        response, extracted_info = extract_info_chatgpt(paper_chunk, questions, prev_dict)
        responses.append({'i':i, 'response':response, 'chunk':paper_chunk, 'extracted_info':extracted_info})
        try:
            prev_dict = eval(extracted_info)
        except:
            prev_dict = extracted_info
        steps.append(prev_dict)
    final_memory = steps[-1]
    return final_memory, steps, responses

if __name__ == "__main__":
    pdf_path = 'test pdfs//'
    files = [x for x in os.listdir(pdf_path) if 'proper_' in x]

    thegap = 8000
    questions = ['What is the research object of this article',
                 'What is (are) the country/region of the study',
                 'What is the data sample size (or observations or n)',
                 'What is the theory name of this paper',
                 'What is the study period of this paper (year or year range)',
                 'Is this paper a qualitative or quantitative study',
                 'How many authors are there in this paper',
                 'What are the main findings for this paper',
                 'What is (are) the methodology name(s) of this paper (or the type of regression etc)',
                 'Is this paper an original study or literature review',
                 'Is this chunk looks like reference list? Answer this question with "ENDENDEND" if so else leave it blank as ""']

    start_time = time.time()
    file_steps = {}
    for file in files:
        text = read_pdf(pdf_path+file)
        text = text.replace('\n','').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').strip()
        final_memory, steps, responses = extract_info_from_text(text, questions, thegap)
        result_dict = {'final_memory':final_memory, 'steps':steps, 'responses':responses}
        file_steps[file] = result_dict
        print(file, '\t', f'cost {time.time() - start_time} seconds')
    end_time = time.time()
    print(f'Cost time {end_time - start_time} seconds')