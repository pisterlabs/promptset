import langchain
from langchain.document_loaders import PyPDFLoader, MathpixPDFLoader, ArxivLoader, PDFMinerPDFasHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
import os
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
import jsonlines
import openai
from tqdm import tqdm
import time

def logger(obj, logger_path):
    # 创建logger文件夹（如果不存在）
    if not os.path.exists('logger'):
        os.makedirs('logger')

    # 获取当前时间戳
    timestamp = str(int(time.time()))

    # 构建日志文件路径
    log_file = logger_path#f'{timestamp}.jsonl')

    # 写入日志内容（追加模式）
    with jsonlines.open(log_file, 'a') as file:
        file.write(obj)

def download_arxiv_paper(arxiv_id, target_folder):
    base_url = 'https://arxiv.org'
    page_url = f'{base_url}/abs/{arxiv_id}'
    #response = requests.get(page_url)

    # Find the PDF link
    pdf_link = f'{base_url}/pdf/{arxiv_id}'

    # Create the final download url
    download_url = pdf_link

    # Download the PDF
    response = requests.get(download_url)

    # Write the PDF to a file
    with open(os.path.join(target_folder, f'{arxiv_id}.pdf'), 'wb') as f:
        f.write(response.content)


def get_summary(nodes, logger_path, model = "gpt-3.5-turbo-16k-0613"):
    ## load the results
    done_dict = {}

    if not os.path.exists(logger_path):
        ## create the logger file if not exists
        with open(logger_path, 'w') as writer:
            writer.write("")


    with jsonlines.open(logger_path ) as reader:
        for obj in reader:
            done_dict[obj['text']] = obj['response']


    prompt_title = """你需要从下面的文本中总结题目、作者、机构，用markdown格式，并翻译为中文： \n\n{text}"""
    if model == "gpt-3.5-turbo-16k-0613":
        prompt_template = """你是一个逻辑严谨、具有深度机器学习知识背景的科研人员，正在阅读一篇论文，为一次论文分享会做准备，你需要将下面的段落中的关键内容用中文整理出来，分类分点总结，为他人解读这个文章中的核心内容：\n\n {text}"""#"""你是一个逻辑严谨、具有深度机器学习知识背景的科研人员，你正在review一篇论文，首先将下面段落中有等有信息量的关键内容总结成中文，分类分点总结，开头用一句话概括：\n\n{text}"""#"""你是一个逻辑严谨、具有深度机器学习知识背景的科研人员，你正在review一篇论文，首先将下面段落中有关核心方法、insight、实验结果、关键结论等有信息量的内容总结成中文，分点总结，开头用一句话概括：\n\n{text}"""
    else:
        prompt_template = """你是一个逻辑严谨、具有深度机器学习知识背景的科研人员，正在阅读一篇论文，为一次论文分享会做准备，你需要将下面的段落中的关键内容用中文整理出来，分类分点总结，为他人解读这个文章中的核心内容：\n\n {text}"""#"""你是一个逻辑严谨、具有深度机器学习知识背景的科研人员，你正在review一篇论文，首先将下面段落中有关核心方法、insight、实验结果、关键结论等有信息量的内容总结成中文，分点总结，用markdown格式，开头用一句话概括：\n\n{text}"""
    for section_name in tqdm(nodes):
        section_info = nodes[section_name]
        for node in section_info['paragraphs']:
            if section_name == 'title':
                prompt = prompt_title.replace("{text}", node)
            else:
                prompt = prompt_template.replace("{text}", node)
            if prompt not in done_dict:
                try:
                    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo-16k-0613",
                      messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                       temperature=0,
                       max_tokens=10000,
                    )
                except:
                    print("error: time too short for server")
                    continue
                time.sleep(10)
                logger({"text": prompt, "response": response.choices[0].message.content, "origin_text": node}, logger_path)
                response_str = response.choices[0].message.content
            else:
                response_str = done_dict[prompt]

            if 'summary' not in section_info:
                section_info['summary'] = ""
            section_info['summary'] = section_info['summary'] + response_str + "\n\n"
    return nodes

def parse_html_to_snippnets(data):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data[0].page_content, 'html.parser')
    content = soup.find_all('div')

    import re

    cur_fs = None
    cur_text = ''
    snippets = []  # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px', st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text, cur_fs))
    return snippets

def generate_summary(paper_list, model):
    for paper_id in paper_list:
        if 'pdf' not in paper_id:
            ## download arvix paper if not exist
            if not os.path.exists(f"../data/paper/{paper_id}.pdf"):
                download_arxiv_paper(paper_id, "../data/paper")

            ## read a pdf text by langchain
            loader = PDFMinerPDFasHTMLLoader(f"../data/paper/{paper_id}.pdf")
        else:
            loader = PDFMinerPDFasHTMLLoader(f"../data/paper/{paper_id}")

        data = loader.load()


        ## get the detail snip
        snippets = parse_html_to_snippnets(data)

        # count the cur_fs times and total str len
        font_dict  = {}
        for snippet in snippets:
            if snippet[1] not in font_dict:
                font_dict[snippet[1]] = [0, 0]
            font_dict[snippet[1]][0] += 1
            font_dict[snippet[1]][1] += len(snippet[0])

        # count avg len
        for key in font_dict:
            font_dict[key].append(font_dict[key][1]/font_dict[key][0])

        # sort dict
        sorted_dict = {key: font_dict[key] for key in sorted(font_dict.keys(), reverse=True)}

        # find the large font_size which satisfy the times > 8, 8<avg_len<300
        section_font_size = 0
        for key in sorted_dict:
            if sorted_dict[key][0] > 6 and sorted_dict[key][2] > 8 and sorted_dict[key][2] < 300:
                section_font_size = key
                break
        print(section_font_size)

        # merge the snippets in order, split by the section_font_size snippet as a section
        section_text = {'title': ""}
        section_title = 'title'
        title_font_size = 0
        for snippet in snippets:
            if section_title == 'title':
                # find max font_size snippet as title
                title_text = ''
                if snippet[1] > title_font_size:
                    title_font_size = snippet[1]
                    title_text = snippet[0]

            if snippet[1] == section_font_size:
                section_title = snippet[0]
                if section_title not in section_text:
                    section_text[section_title] = ''
            else:
                section_text[section_title] += snippet[0]

        # Tokenize the text into sentences for each section
        for key in section_text:
            sentences = sent_tokenize(section_text[key])
            section_text[key] = {'sentences': sentences, 'text': section_text[key]}
            # remove meaningless \n in each sentence
            for i in range(len(sentences)):
                sentences[i] = sentences[i].replace('\n', ' ')

        ## save the text into a markdown file
        with open(f"../data/paper/{paper_id}.md", "w") as f:
            for key in section_text:
                f.write(f"# {key}\n")
                for sentence in section_text[key]['sentences']:
                    f.write(f"{sentence}\n")
                f.write("\n")

        # merge sentences into paragraphs chunk, and each chunk with in 4000 tokens by tiktoc
        for key in section_text:
            section_text[key]['paragraphs'] = []
            section_text[key]['paragraphs_token_len'] = []
            section_text[key]['paragraphs'].append(section_text[key]['sentences'][0])
            section_text[key]['paragraphs_token_len'].append(len(word_tokenize(section_text[key]['sentences'][0])))
            for i in range(1, len(section_text[key]['sentences'])):
                if len(word_tokenize(section_text[key]['paragraphs'][-1])) + len(word_tokenize(section_text[key]['sentences'][i])) < 4000:
                    section_text[key]['paragraphs'][-1] += ' ' + section_text[key]['sentences'][i]
                else:
                    section_text[key]['paragraphs_token_len'].append(len(word_tokenize(section_text[key]['paragraphs'][-1])))
                    section_text[key]['paragraphs'].append(section_text[key]['sentences'][i])
            section_text[key]['paragraphs_token_len'].append(len(word_tokenize(section_text[key]['paragraphs'][-1])))

        # summary information
        summary = get_summary(section_text, f"../logger/paper/{paper_id}_summary_{model}.md")

        ## save the summary into a markdown file
        with open(f"../data/paper/{title_text}_{paper_id}_summary_{model}.md", "w") as f:
            for key in summary:
                f.write(f"# {key}\n")
                try:
                    f.write(f"{summary[key]['summary']}\n")
                except:
                    pass
                f.write("\n")
        pass

if __name__ == '__main__':
    paper_list = ["2307.04349", "2307.03109"] #also support use "../data/paper/2307.04349.pdf
    model = "gpt-3.5-turbo-16k-0613"  # ["gpt-3.5-turbo-16k-0613", "gpt-4"]
    generate_summary(paper_list, model)