from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import DataFrameLoader, PyMuPDFLoader


import os
import fitz
import pandas as pd
import json
import ast

def fonts(doc, granularity=False, pages=2):
    """Extracts fonts and their usage in PDF documents.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool

    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}
    pageCounter = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage
        pageCounter += 1
        if pageCounter >= pages:
            break

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles


def font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.

    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict

    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag


def get_pdf_raw_pages(doc, pages):
    header_para = []
    pageCounter = 0
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        header_para.append(blocks)
        pageCounter += 1
        if pageCounter >= pages:
            break
    return header_para


def headers_para(doc, size_tag, pages=2):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict

    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    pageCounter = 0
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            # header_para.append("<section_block>")
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                            else:
                                if s['size'] == previous_s['size']:

                                    if block_string and all((c == "|") for c in block_string):
                                        # block_string only contains pipes
                                        block_string = size_tag[s['size']] + s['text']
                                    if block_string == "":
                                        # new block has started, so append size tag
                                        block_string = size_tag[s['size']] + s['text']
                                    else:  # in the same block, so concatenate strings
                                        block_string += " " + s['text']

                                else:
                                    header_para.append(block_string)
                                    block_string = size_tag[s['size']] + s['text']

                                previous_s = s

                    # new block started, indicating with a pipe
                    block_string += "|"

                # header_para.append("<text_block>")
                header_para.append(block_string)
                # header_para.append("<text_block_end>")

            # header_para.append("<section_block_end>")

        pageCounter += 1
        if pageCounter >= pages:
            break

    return header_para


def get_pdf_first_page_txt(pdf_path, pages=2):
    doc = fitz.open(pdf_path)

    font_counts, styles = fonts(doc, granularity=False, pages=pages)

    size_tag = font_tags(font_counts, styles)

    return headers_para(doc, size_tag, pages)

def get_pdf_pages(pdf_path, pages=2):
    docs = PyMuPDFLoader(pdf_path).load()
    return docs[:pages]
    # texts = []
    # for doc in docs[:pages]:
    #     texts.append(doc.page_content)
    # return texts

def get_pdf_page_metadata(pdf_path, pages):
    pdf_first_page_txt = get_pdf_first_page_txt(pdf_path, pages)

    template = """
                I have extracted text from the initial pages of a Journal of Economic Literature (JEL) PDF file. I require assistance in extracting 
                specific details, namely: article title, author, abstract and keywords section. Please be aware that if you encounter 
                JEL classifications such as C12 and P34, kindly ignore them and refrain from including them in the abstract and keywords.                
                
                {format_instructions}

                Wrap your final output as a json objects

                INPUT:
                {pdf_first_page_txt}

                YOUR RESPONSE:
    """
    response_schemas = [
        ResponseSchema(name="title", description="extracted title"),
        ResponseSchema(name="author", description="extracted authors seperated by comma"),
        ResponseSchema(name="abstract", description="extracted abstract"),
        ResponseSchema(name="keywords", description="extracted keywords")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(template)  
        ],
        input_variables=["pdf_first_page_txt"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.0,max_tokens=6048) # type: ignore gpt-3.5-turbo

    final_prompt = prompt.format_prompt(pdf_first_page_txt=pdf_first_page_txt)
    output = llm(final_prompt.to_messages())

    try:
        result = output_parser.parse(output.content)
    except:
        if "```json" in output.content:
            json_string = output.content.split("```json")[1].strip()
        else:
            json_string = output.content
        result = fix_JSON(json_string)

    head, tail = os.path.split(pdf_path)

    result["filename"] = tail

    return result

def get_pdf_page_accept_metadata(pdf_path, pages):
    pdf_first_page_txt = get_pdf_first_page_txt(pdf_path, pages)

    template = """
                I have extracted text from the initial pages of a Journal of Economic Literature (JEL) PDF file. 
                I need help identifying the accepted date of the article. If the accepted date is not explicitly specified, 
                it should be located either at the top or bottom of the first or second page of the article in a date format without the prefix 'accepted'.                
                
                {format_instructions}

                Wrap your final output as a json objects

                INPUT:
                {pdf_first_page_txt}

                YOUR RESPONSE:
    """
    response_schemas = [
        ResponseSchema(name="accepted", description="extracted accepted date")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(template)  
        ],
        input_variables=["pdf_first_page_txt"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.0,max_tokens=148) # type: ignore gpt-3.5-turbo

    final_prompt = prompt.format_prompt(pdf_first_page_txt=pdf_first_page_txt)
    output = llm(final_prompt.to_messages())

    try:
        result = output_parser.parse(output.content)
    except:
        if "```json" in output.content:
            json_string = output.content.split("```json")[1].strip()
        else:
            json_string = output.content
        result = fix_JSON(json_string)

    head, tail = os.path.split(pdf_path)

    result["filename"] = tail

    return result

def get_pdf_intro(pdf_path, pages):
    pdf_first_page_txt = get_pdf_first_page_txt(pdf_path, pages)

    template = """
                I have extracted text from the initial pages of a Journal of Economic Literature (JEL) PDF file. I require assistance in extracting 
                introduction section.  Typically, the introduction section begins after the abstract and ends before the next sub-title or section heading.         
                
                Wrap your final output as a json objects

                INPUT:
                {pdf_first_page_txt}

                YOUR RESPONSE:
    """
    response_schemas = [
        ResponseSchema(name="introduction", description="extracted introduction")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(template)  
        ],
        input_variables=["pdf_first_page_txt"],
        # partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.0,max_tokens=8396) # type: ignore gpt-3.5-turbo

    final_prompt = prompt.format_prompt(pdf_first_page_txt=pdf_first_page_txt)
    output = llm(final_prompt.to_messages())

    try:
        result = output_parser.parse(output.content)
    except Exception as e:
        print(str(e))
        if "```json" in output.content:
            json_string = output.content.split("```json")[1].strip()
        else:
            json_string = output.content
        result = fix_JSON(json_string)

    head, tail = os.path.split(pdf_path)

    result["filename"] = tail

    return result


def get_polish_intro(my_intro, sample_introes, words_limit, temperature):
    template = """
                I require an introduction for my Journal of Economic Literature and I would appreciate it \
                if you could compose it for me around {words_limit} words. I would like the introduction mimic on the \
                sample introductions that I will provide. If I have already provided my own introduction, \
                please refine it accordingly. 

                % My own introduction: {my_intro}

                % Sample introductions:
                {sample_introes}
                % End of sample introductions:

                YOUR RESPONSE:
    """
    response_schemas = [
        ResponseSchema(name="introduction", description="refined introduction")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(template)  
        ],
        input_variables=["my_intro","sample_introes","words_limit"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=temperature,max_tokens=2048) # type: ignore gpt-3.5-turbo

    final_prompt = prompt.format_prompt(my_intro=my_intro, sample_introes=sample_introes, words_limit=words_limit)
    output = llm(final_prompt.to_messages())

    result = output.content

    return result


def fix_JSON(json_message=None):
    result = None
    try:        
        result = json.loads(json_message)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')', ''))        
        # Remove the offending character:
        json_message = list(json_message)
        json_message[idx_to_replace] = ' '
        new_message = ''.join(json_message)     
        return fix_JSON(json_message=new_message)
    return result


def save_pdfs_to_db(pdf_files, excel_file, meta_type='meta', pages=2):
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        existing_data = df.to_dict(orient='records')
    else:
        existing_data = []

    existing_filenames = set(row['filename'] for row in existing_data)

    
    for doc in pdf_files:
        head, tail = os.path.split(doc)
        if tail not in existing_filenames:
            # print('get meta from LLM '+doc)
            try:
                if meta_type == 'intro':
                    metadata = get_pdf_intro2(doc, pages)
                elif meta_type == 'date':
                    metadata = get_pdf_page_accept_metadata(doc, pages)
                else:
                    metadata = get_pdf_page_metadata(doc, pages)

                temp_data = []
                temp_data.append(metadata)
                save_to_excel(existing_data+temp_data, excel_file)
                existing_data += temp_data
                print("Data append to ", excel_file)
            except Exception as e:
                print(str(e))


def get_metadata_from_db(excel_file):
    df = pd.read_excel(excel_file)
    dict = df.to_dict(orient='records',)
    return dict


def get_column_from_db(excel_file, column):
    df = pd.read_excel(excel_file)
    doc = DataFrameLoader(df, column).load()
    return doc


def get_data_from_csv(file_path, column_name, filter_value):
    data = pd.read_csv(file_path, encoding = 'unicode_escape')
    filtered_data = data[data[column_name] == filter_value]
    dict_data = filtered_data.to_dict(orient='records') #filtered_data.values.tolist()
    for row in dict_data:
        md = ast.literal_eval(row["metadata"])
        # print(type(md))
        row["date"] = md["modDate"]
    return dict_data


def get_filename_list(similar_dict, path):
    filenames = []
    for doc in similar_dict['context']:
        filenames.append(os.path.join(path, doc.metadata['filename']))
    return filenames


def save_to_excel(data, file_path):
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    

def get_pdf_intro2(pdf_path, pages):
    pdf_first_page_txt = get_pdf_first_page_txt(pdf_path, pages)
    # pdf_first_page_txt = get_pdf_pages(pdf_path, pages)

    human_template = """
I have extracted the text from the initial pages of a Journal of Economic Literature (JEL) PDF file. I require assistance in extracting the introduction section. Typically, the document follows a pattern where the 'abstract' header is encountered, followed by the abstract section. Subsequently, an 'Introduction' header is expected, which is followed by the introduction section. Next, there may be a 'Background' header or other headers indicating different sections. The introduction section generally concludes before the next sub-title or section heading appears, such as 'Background' or other similar headings.

Please continue searching for the introduction section until you reach a clear next sub-title or section heading. However, please note that if you encounter a bottom part between two pages, such as a section starting with 'RECEIVED:' followed by a date, it does not necessarily mean that the introduction section has ended. In such cases, you should continue searching on the next page.

If the text 'www.elsevier.com' appears in the beginning, it indicates that the literature is published on Elsevier and follows a specific format. In this case, the abstract section will start with "A B S T R A C T" and end before the introduction section. The introduction section will typically start with "1. Introduction" and end before the next section header, such as "2. Background". Please continue searching for the introduction section until you reach next section heading such as "2. Background", it has to be started with "2.".

Please provide the introduction section as the final output in JSON format with the key 'Introduction' written in Pascal case.

Exclude the content of the abstract section.

Only include the text within the introduction section and exclude any text prior to it.

INPUT: {pdf_first_page_txt}

YOUR RESPONSE:
    """
    response_schemas = [
        # ResponseSchema(name="abstract", description="extracted abstract"),
        ResponseSchema(name="introduction", description="extracted introduction")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(human_template)  
        ],
        input_variables=["pdf_first_page_txt"]
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.0,max_tokens=6658) # type: ignore gpt-3.5-turbo

    final_prompt = prompt.format_prompt(pdf_first_page_txt=pdf_first_page_txt)
    output = llm(final_prompt.to_messages())

    try:
        result = output_parser.parse(output.content)
    except Exception as e:
        print(str(e))
        if "```json" in output.content:
            json_string = output.content.split("```json")[1].strip()
        else:
            json_string = output.content
        result = fix_JSON(json_string)

    head, tail = os.path.split(pdf_path)

    result["filename"] = tail

    return result


def main():
    
    documents = ['./data/docs/literature/Do people care about democracy_An experiment exploring the value of voting rights.pdf',
                    './data/docs/literature/Expressive voting versus information avoidance_expenrimental evidence in the context of climate change mitigation.pdf',
                    './data/docs/literature/Crashing the party_An experimental investigation of strategic voting in primary elections.pdf',
                    './data/docs/literature/Economic growth and political extremism.pdf']  
    doc = './data/docs/literature_suicide/1-s2.0-S0304387821000432-main.pdf'
    doc = './data/docs/literature_suicide/1-s2.0-S0047272721000761-main.pdf'
    # doc = './data/docs/literature_suicide/rest_a_00777.pdf'
    documents = ['./data/docs/literature/Do people care about democracy_An experiment exploring the value of voting rights.pdf'
                  ,'./data/docs/literature/Expressive voting versus information avoidance_expenrimental evidence in the context of climate change mitigation.pdf'
                   ,'./data/docs/literature/Economic growth and political extremism.pdf'   ]
    # './data/docs/literature/Expressive voting versus information avoidance_expenrimental evidence in the context of climate change mitigation.pdf',
    #                 './data/docs/literature/Crashing the party_An experimental investigation of strategic voting in primary elections.pdf',
    #                 './data/docs/literature/Economic growth and political extremism.pdf']  
   
    # save_pdfs_to_db(documents, intro_excel_file, is_Intro=True, pages=4)
    metadata = get_pdf_intro2(doc, 2)
    print(metadata)
    # docs = get_pdf_first_page_txt(doc, 3)
    # # docs = get_pdf_pages(doc, 2)
    # # docs = get_pdf_raw_pages(doc, 2)
    # print(docs)
    # pdf_first_page_txt = get_pdf_first_page_txt(doc, 3)
    # raw_txt = get_pdf_raw_pages(fitz.open(doc), 2)
    # print(raw_txt)
    # pdf_first_page_txt = get_pdf_first_page_txt(doc, 3)

    # output_file = "data/db/repo_intro_4.xlsx"

    # intro354_excel_file = "data/db/repo_intro_35_16.xlsx"
    # save_pdfs_to_db(documents, intro354_excel_file, is_intro=True, pages=4)

    # intros = [dict["introduction"] for dict in get_metadata_from_db(intro35_excel_file)]
    # polish = get_polish_intro('', intros[:3], 600, 0)
    # print(polish)


    # csv_file = "./data/db/summary.csv"
    # column_name = "Theme"
    # filter_value = "China authoritarian system"
    # data = get_data_from_csv(csv_file, column_name, filter_value)
    # print(data)
    

if __name__ == '__main__':
    main()