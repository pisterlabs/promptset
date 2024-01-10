import re

import pandas as pd
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from load_documents import load_documents, load_split_documents
from prompt_templates import prompt_paper_sum, prompt_sum_translate


def summarize_papers(uploaded_files_path, llm=None, language='en', auto_language=False, verbose=False, print_response=True):
    rows = []
    documents = load_documents(uploaded_files_path)
    documents_spilt = load_split_documents(uploaded_files_path)

    for i in range(len(documents)):
        try:
            paper = documents[i]
            response = chain_paper_sum_stuff(paper, llm=llm, language=language, auto_language=auto_language,
                                             verbose=verbose, print_response=print_response)
        except:
            paper = [documents_spilt[i][0]]
            response = chain_paper_sum_stuff(paper, llm=llm, language=language, auto_language=auto_language,
                                             verbose=verbose, print_response=print_response)

        row = format_paper_sum_response(response)
        rows.append(row)

    df = pd.DataFrame(rows)
    df['Conclusion'] = df['Conclusion'].replace("   ", "\n")

    return df


def chain_paper_sum_stuff(document, llm=None, language='en', auto_language=False, verbose=False, print_response=True):
    if auto_language:
        language = check_language_cn(document[0].page_content)

    # Define prompt
    prompt = prompt_paper_sum.get(language)
    prompt = PromptTemplate.from_template(prompt)
    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text", verbose=verbose
    )

    response = stuff_chain.run(document)

    if print_response:
        print(response)

    return response


def format_paper_sum_response(response):
    language = check_language_cn(response)
    if language == 'en':
        target = "6. Conclusion:"
        parts = ["1. Title:", "2. Authors:", "3. Research Topics:",
                 "4. Research Method:", "5. Data Source:", "6. Conclusion:"]
    elif language == 'cn':
        target = "6. 结论:"
        parts = ["1. 题目:", "2. 作者:", "3. 研究主题:", "4. 研究方法:", "5. 数据来源:", "6. 结论:"]

    index = response.find(target)
    replacement = "   "
    if index != -1:
        response_split = response[:(index + len(target))] + response[(index + len(target)):].replace("\n", replacement)
        for part in parts:
            response_split = response_split.replace(part, "")
        response_split = [result.strip() for result in response_split.split("\n")]
    else:
        ValueError(f"Could not find target in response. Response format error:\n{response}")

    response_dict = {
        "Title": response_split[0],
        "Authors": response_split[1],
        "Research Topic": response_split[2],
        "Research Method": response_split[3],
        "Data Source": response_split[4],
        "Conclusion": response_split[5].replace(replacement, "\n")
    }

    return response_dict


def chain_translate_sum(response, llm=None, language='en'):
    prompt = prompt_sum_translate.get(language)

    prompt_translate = PromptTemplate.from_template(prompt)
    chain_translate = LLMChain(llm=llm, prompt=prompt_translate)
    response_translated = chain_translate.run(response)

    return response_translated


def check_language_cn(text):
    text_cn = re.findall(r'[\u4e00-\u9fff]+', text)
    if len(text_cn) / len(text) > 0.05:
        language = "cn"
    else:
        language = "en"

    return language
