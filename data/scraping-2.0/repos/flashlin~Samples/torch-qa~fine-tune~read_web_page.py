import json
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_lit import load_markdown_documents
from web_crawler_lit import download_html, convert_html_body_to_markdown
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


def load_llm_model(model_name):
    return LlamaCpp(
        model_path=model_name,
        temperature=0.75,
        max_tokens=2000,
        top_p=3,
        n_ctx=1024 * 16,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=True,
        n_gpu_layers=52,
        n_threads=16,
    )

llm = None

def ask_content(content: str, instruct: str):
    global llm
    instruct = instruct.strip()
    prompt = f"""
    [Content]
    {content}
    [/Content]
    {instruct}
    """
    return llm(prompt)

def rewrite_content(content: str):

    content = """
# 8% SBO Live Baccarat Weekly Cashback

__12245 __Fri, Nov 17, 2023 __[Promotions Terms & Conditions](https://help.sbotop.com/category/rules-regulations/promotions-terms-conditions/35/ "Categories » Rules & Regulations » Promotions Terms & Conditions ")

** [ Take Part!  ](https://account.sbotop.com/register?lg=en) **

_ **SBO Live Baccarat 8% Cashback up to RM 500 every week!**_

Promotion Period: 20th November 2023 12:00 PM (GMT+8) to 1st January 2024 11:59 AM (GMT+8) (“Promotion Period”).

###  How it works? 

  1. Opt in with Mobile and Bet on SBO Live Baccarat during promotion period.
  2. Place a minimum of 8 rounds of bets on SBO Live Baccarat to be counted.   


  3. You will receive an 8% Cashback Bonus from your Total Net Loss up to RM 500 or currency equivalent.



###  Conditions: 

  1. Promotion will be commenced from 20th November 2023 12:00 PM (GMT+8) to 1st January 2024 11:59 AM (GMT+8) (“Promotion Period”).
  2. This promotion is offered to CN¥, RM, Rp, THB (฿), MMK, JP¥, KRW (₩), VND (₫), R$ & USDT currency holders only.
  3. Bets a minimum of 8 rounds on SBO Live Baccarat during promotion period to be counted in the Total Net Loss calculation. Any bet placed before or after the promotion period will not be counted to the Total Net Loss calculation.
  4. Cashback bonus will be credited every Tuesday 18:00 (GMT+8).
  5. Any FREE BET, CANCELLED, DRAW, REJECT, REFUND, and VOID bet will not be counted towards the Total Net Loss calculation.
  6. The maximum cashback bonus amount is RM 500.
  7. The minimum cashback bonus amount is RM 8. If the cashback bonus amount is less than RM 8, no bonus shall be credited.
  8. Bonus with decimal amount will be round up to the nearest amount.
  9. The bonus will be credited in the form of SBO Live Baccarat free bet voucher and only can be used via SBOTOP Mobile or App Version. SBOTOP Voucher Terms and Conditions apply.
  10. No rollover requirement of the winnings from the voucher.
  11. SBOTOP reserves the right to amend and/or cancel this promotion anytime.  
  

  12. [General Promotion Terms and Conditions apply.](https://help.sbotop.com/article/general-promotion-terms-conditions-265.html)   
  

  13. [Live Casino Gaming Rules apply.](https://help.sbotop.com/secure/live-casino-betting-rules-755.html)   
  

  14. [General Terms and Conditions apply.](https://help.sbotop.com/article/general-promotion-terms-conditions-265.html)   
  




### Index of Promotion: 

  1. Maximum cashback bonus for currencies equivalent to RM 500:  
  


RM | Rp | THB (฿) |  VND (₫) | CN¥ | MMK | JP¥ | USDT | KRW (₩) | R$  
---|---|---|---|---|---|---|---|---|---  
500 | 1,688,888 | 3,888 | 2,688,888 | 688 | 288,000 | 15,888 | 108 | 148,888 | 500  
  
  2. Minimum cashback bonus for currencies equivalent to RM 8:  
  


RM | Rp | THB (฿) |  VND (₫) | CN¥ | MMK | JP¥ | USDT | KRW (₩) | R$  
---|---|---|---|---|---|---|---|---|---  
8 | 28,000 | 58 | 48,000 | 18 | 5,000 | 248 | 1 | 2,800 | 8  
    """.strip()

    return ask_content(content, """Read the markdown content above, 
identify entities, such as keywords, title, subject, people, events, currencies, countries, things, date, numbers, and times. 
if you encounter content that you cannot understand, skip the part of the content that you cannot understand. 
Retain the original markdown content's titles and chapter structure. 
The article content generated must have titles and chapters.
Please modify the original markdown content and send it to me directly.
    """)

def generate_questions_from_markdown(content: str):
    global llm
    prompt_template = """
    ```
    {content}
    ```
    Read the content above, identify entities, such as keywords, title, subject, people, events, things, date, numbers, and times.
    Generate 10 different questions based on the content about these entities, avoiding repetition of the same entity. 
    List the 10 english questions directly.
    """
    prompt = prompt_template.format(content=content)
    questions_content = llm(prompt)
    return questions_content


def get_answer_from_content(content: str, question: str):
    global llm
    prompt_template = """
    ```
    {content}
    ```
    Read the content above, My Question is `{question}`
    Answer the questions based on the content above. 
    If the answer is not found in the content, try to think of an answer yourself. 
    If you do not know the answer, simply answer `None`. Do not try to create false answers.
    """
    prompt = prompt_template.format(content=content, question=question)
    answer = llm(prompt)
    return answer


def extrac_question_body(question_line: str):
    match = re.match(f'\d+. (.*)', question_line)
    if match:
        return match.group(1).strip()
    return None


def split_questions_content(content: str):
    question_lines = re.findall(r'\d+\. .*', content)
    questions = []
    for question_line in question_lines:
        q = extrac_question_body(question_line)
        questions.append(q)
    return questions


def append_to_jsonl(question: str, answer: str):
    with open('./results/llm-qa.jsonl', 'a', encoding='utf-8') as f:
        qa_json = json.dumps({
            'instruction': question,
            'input': '',
            'output': answer
        })
        f.write(qa_json + '\r\n')


def append_to_md(question: str, answer: str):
    with open('./results/llm-qa.md', 'a', encoding='utf-8') as f:
        f.write(f'Question: {question}\r\n')
        f.write(f'Answer: {answer}\r\n')

def remove_bad_content(content: str) -> str:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('** [ Take Part!'):
            lines[i] = ''
    new_content = '\n'.join(lines)
    return new_content

def clean_content(content: str) -> str:
    index = content.find("This article is also available in the following languages:")
    if index != -1:
        content = content[:index]
    index = content.find("# Feedback")
    if index != -1:
        content = content[:index]
    index = content.find("**View in other languages")
    if index != -1:
        content = content[:index]
    index = content.find("**View other languages")
    if index != -1:
        content = content[:index]
    content = remove_bad_content(content)
    return content


def clean_file(file: str):
    with open(file, "r", encoding='utf-8') as f:
        content = f.read()
    new_content = clean_content(content)
    if len(new_content) != len(content):
        print(f"{file} clean done.")
        with open(file, "w", encoding='utf-8') as f:
            f.write(new_content)

def clean_files(folder: str):
    file_names = os.listdir(folder)
    for file_name in file_names:
        clean_file(f"{folder}/{file_name}")


if __name__ == '__main__':
    clean_files('./data')

    print("loading model...")
    model_name = "neural-chat-7b-v3-16k.Q4_K_M"
    model_name = "orca-2-13b.Q4_K_S"
    llm = load_llm_model(f'../models/{model_name}.gguf')
    # html = download_html('https://ithelp.ithome.com.tw/articles/10335513')
    # markdown = convert_html_body_to_markdown(html)

    # new_markdown = rewrite_content("")
    # with open(f'./results/llm_rewrite.md', 'w', encoding='utf-8') as f:
    #     f.write(new_markdown)

    print("load documents...")
    documents = load_markdown_documents('./data')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 * 10, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    for doc in all_splits:
        markdown = doc.page_content
        source = doc.metadata['source']

        # markdown = """
        # PageAttention’s memory sharing greatly reduces the memory overhead of complex sampling algorithms, such as parallel sampling and beam search, cutting their memory usage by up to 55%. This can translate into up to 2.2x improvement in throughput. This makes such sampling methods practical in LLM services.
        # PagedAttention is the core technology behind vLLM, our LLM inference and serving engine that supports a variety of models with high performance and an easy-to-use interface. For more technical details about vLLM and PagedAttention, check out our GitHub repo and stay tuned for our paper.
        # """

        # print(f"rewrite document for {source}")
        # new_markdown = rewrite_content(markdown)
        # with open(f'./results/llm_{file_index}.md', 'w', encoding='utf-8') as f:
        #     f.write(new_markdown)
        # print(f"write llm_{file_index} done")
        # file_index += 1

        print(f"generate questions for {source}")
        questions_content = generate_questions_from_markdown(markdown)
        print(f"{questions_content=}")
        questions = split_questions_content(questions_content)
        for question in questions:
            print(f"ask {question} for {source}")
            answer = get_answer_from_content(markdown, question)
            answer = answer.strip()
            if answer != 'None':
                append_to_jsonl(question, answer)
                append_to_md(question, answer)

        