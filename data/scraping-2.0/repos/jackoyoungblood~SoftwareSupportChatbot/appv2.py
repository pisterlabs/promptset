from PyPDF2 import PdfReader
import os
import re
import openai
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import gradio as gr

nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    result = set()
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if stemmed_token not in stops:
            result.add(stemmed_token)
    return result

def get_toc(reader, toc_page_start, toc_page_end):
    toc_dict = {}
    section = 0
    regex = r'(?P<section>[\w+\s+]+) (?:\.+) (?P<page_num>\d+)'
    contents = reader.pages[toc_page_start - 1:toc_page_end]
    for page in contents:
        page_contents = page.extract_text().split('\n')
        for line in page_contents:
            m = re.search(regex, line)
            if m:
                groupdict = m.groupdict()
                item = {
                    'section_words': preprocess_sentence(groupdict['section'].lower()), 
                    'page_num': groupdict['page_num']
                }
                toc_dict[section] = item
                section += 1
    return toc_dict

def get_page_nums_with_keywords(toc_dict, keywords, page_limit=3):
    max_pages = 0
    results = set()
    for item in toc_dict.values():
        intersect = len(keywords.intersection(item['section_words']))
        union = len(keywords.union(item['section_words']))
        similarity = intersect / union
        results.add((similarity, item['page_num']))
    sections = sorted(list(results), reverse = True)[:page_limit]
    return set([int(section[1]) for section in sections])

def get_page_nums_with_keyword(toc_dict, keyword, page_limit = 3):
    sections = list(
        filter(lambda section: keyword in toc_dict[section]['section_words'],  toc_dict.keys())
    )
    return list(set([int(toc_dict[section]['page_num']) for section in sections]))[:page_limit]

def get_content_from_pages(reader, page_nums):
    content = ""
    for page_num in page_nums:
        content += reader.pages[page_num - 1].extract_text()
    return content

def make_prompt(content, question, add_task_description = False):
    prompt=''
    prompt += content 
    if add_task_description:
        prompt += '\n'
        prompt += 'Answer the question as truthfully as possible and if you are unsure of the answer say "Sorry I don''t know"'
    prompt += '\n'
    prompt += 'Q: '
    prompt += question
    prompt += '\n'
    prompt += 'A:'
    return prompt

def get_openai_response(prompt):
    return openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.0,
        max_tokens=104,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["Q:"]
    )


def predict(input, history=[]):
    response=''

    if input == '':
        return

    if 'license' in input.lower():
        return

    if 'subscription' in input.lower():
        return
    
    if 'ignore the above' in input.lower():
        return

    if 'imanage records manager' not in input.lower():
        input = input.replace("?","")
        input += ' in imanage records manager?'

    isflagged=get_openai_moderation_response(input)

    if isflagged:
        return

    reader = PdfReader("IRM_Web_Client_User_Guide_Legal_Version_10.3.3.pdf")
    toc_dict = get_toc(reader, 4, 8)
    keywords = preprocess_sentence(input)
    page_nums = get_page_nums_with_keywords(toc_dict, keywords)
    content = get_content_from_pages(reader, page_nums)

    prompt = make_prompt(content, input)
    response = get_openai_response(prompt)
    
    responsetext = response['choices'][0]['text']
    responsetext = responsetext.replace('IRM Web Client User Guide (Legal Version)', '')
    responsetext = responsetext.replace('•','')

    for i in range(0, 100):
        responsetext = responsetext.replace(str(i), '')

    history.append((input,responsetext))
    return history, history

def get_openai_moderation_response(texttomoderate):
    """check text for harmful content using OpenAI's Moderation endpoint and return any items from the categories dict that are true"""
    openai.api_key = os.getenv('OPENAIKEY')
    response = openai.Moderation.create(    
        input=texttomoderate,
        model='text-moderation-stable'
    )
    return response['results'][0]['flagged']
    
if __name__ == "__main__":    
    openai.api_key = os.getenv('OPENAIKEY')  
    
    gr.Interface(fn=predict,
             inputs=["text", "state"],
             outputs=["chatbot", "state"],
             examples=[["How do I generate a box detail report?"],
             ["Can you list the steps to printing an individual label in imanage records manager?"],
             ["Why is the reassign context menu on a file part unavailable?"],
             ["What does it mean for a file part to be in a supersedes relationship with another file part?"],
             ["How do I view pending delivery requests in imanage records manager?"],
             ["How do I view an electronic rendition in iManage Records Manager?"]]).launch()
 