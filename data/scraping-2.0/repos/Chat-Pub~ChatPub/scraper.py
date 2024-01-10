"""
The integrated process to crawl the entire data from 
https://www.youthcenter.go.kr/main.do
https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifList.do
https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifDtl.do

The specific process will be determined later.

Process & Description : 
1. Get whole data of young_policies by options. options will be determined by user. 
(e.g. Region, Category, Core keywords)

2. Data structure of crawled data will be defaultdict.

3. We will give YP number to each data.
main data structure will be [YP, title, R-number, contents]

"""


import argparse
import datetime
import logging.handlers
import sys
import numpy as np
import os
import pandas as pd
import pickle
import requests
import ssl
import ipdb
import time
import warnings
import math
import re
import json
import mysql.connector
import openai
import yaml
import shutil

from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from fake_headers import Headers
from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm, trange
from pprint import pprint

TABLEINFO = ['YP_all_overview', 'YP_summary']

parser = argparse.ArgumentParser()

def get_id(conn):

    #ids will count all data in web.
    ids = set()

    err_count=0
    url = 'https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifList.do?'
    url += 'pageIndex=1'
    url += '&srchRegion=003001&srchRegion=003002001&srchRegion=003002008&srchSortOrder=1&pageUnit=12'
    while True:
        try:
            headers = Headers(headers=True).generate()
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            try:
                total_num=int(float(soup.find(class_='result-num').find('span').text))
            except:
                print("result-num error : cannot find result-num")
                break
            pages = math.ceil(total_num/12)

            #get r_number by changing url
            for page_idx in range(2,2+pages):
                headers = Headers(headers=True).generate()
                res = requests.get(url, headers=headers)
                soup = BeautifulSoup(res.text, 'html.parser')
                for index, contents in enumerate(soup.find_all(class_='tit-wrap')):
                    try:
                        #'title','r_number'
                        ids.add((contents.text.replace('\n','').strip(),contents.a.get('id')[8:]))
                    except:
                        continue
                url = 'https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifList.do?'
                url += 'pageIndex='
                url += f'{page_idx}'
                url += '&srchRegion=003001&srchRegion=003002001&srchRegion=003002008&srchSortOrder=1&pageUnit=12'
            break
        except:
            err_count += 1
            if err_count > 10:  
                print("error")
                break
            time.sleep(1)
            continue

    #fetch data from database
    cursor = conn.cursor()
    sql = "SELECT title, r_number FROM yp_all_overview"
    cursor.execute(sql)
    data_in_db = set(cursor.fetchall())

    #new policies
    data_to_be_added = ids - data_in_db
    
    print(f"There are new {len(data_to_be_added)} policies.")

    #final dataset, origin dataset(could be updated) + new dataset(should be added)
    data_final = sorted(list(data_in_db)) + sorted(list(data_to_be_added))
    
    ret_list=[]
    for index, contents in enumerate(data_final):
        contents=list(contents)
        url = "https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifDtl.do?pageIndex=1&bizId="
        #add R-number
        url += contents[1]
        contents.append(index)
        contents.append(url)
        ret_list.append(contents)

    return ret_list

def doc_parser(contents):
    headers = Headers(headers=True).generate()
    url = contents[3]
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    main_div = soup.find('div', id='wrap')
    
    if main_div is None:
        return None
    
    Tag_list = main_div.find("div", class_='policy-detail')
    return_dict = defaultdict()

    '''
    The codes below are process of extracting data field
    data will be stored in return_dict

    replace " to ' handle sql insertion error
    '''

    #main title
    main_title = Tag_list.find("h2").text
    main_title = main_title.replace("\"","'")

    #short description
    short_description = Tag_list.find(class_="doc_desc").text
    short_description = short_description.replace("\"","'")

    #SUMMARY dictionary
    summary_dict = defaultdict()
    
    summary_contents = Tag_list.find_all(class_="table_wrap")[0].find("ul")
    for index, contents in enumerate(summary_contents.find_all("li")):
        key = re.sub(r'[\r\n\t]', '',contents.find(class_="list_tit").text).strip()
        value = contents.find(class_="list_cont").find_all(string=True)
        if len(value)==0:
            value = None
        elif len(value)==1:
            value = re.sub(r'[\r\n\t]', '',value[0].text).strip()
        else:
            newvalue=""
            for v in value:
                v = re.sub(r'[\r\n\t]', '',v.text).strip()
                newvalue += v
                if v != '':
                    newvalue += '\n'
            value = newvalue.strip('\n')
        value = value.replace("\"","'")
        summary_dict[key] = value
    
    #QUALIFICATION dictionary
    qualification_dict = defaultdict()
    
    qualification_contents = Tag_list.find_all(class_="table_wrap")[1].find("ul")
    for index, contents in enumerate(qualification_contents.find_all("li")):
        key = re.sub(r'[\r\n\t]', '',contents.find(class_="list_tit").text).strip()
        value = contents.find(class_="list_cont").find_all(string=True)
        if len(value)==0:
            value = None
        elif len(value)==1:
            value = re.sub(r'[\r\n\t]', '',value[0].text).strip()
        else:
            newvalue=""
            for v in value:
                v = re.sub(r'[\r\n\t]', '',v.text).strip()
                newvalue += v
                if v != '':
                    newvalue += '\n'
            value = newvalue.strip('\n')
        value = value.replace("\"","'")
        qualification_dict[key] = value

    #METHODS dictionary
    methods_dict = defaultdict()
    
    methods_contents = Tag_list.find_all(class_="table_wrap")[2].find("ul")
    for index, contents in enumerate(methods_contents.find_all("li")):
        key = re.sub(r'[\r\n\t]', '',contents.find(class_="list_tit").text).strip()
        value = contents.find(class_="list_cont").find_all(string=True)
        if len(value)==0:
            value = None
        elif len(value)==1:
            value = re.sub(r'[\r\n\t]', '',value[0].text).strip()
        else:
            newvalue=""
            for v in value:
                v = re.sub(r'[\r\n\t]', '',v.text).strip()
                newvalue += v
                if v != '':
                    newvalue += '\n'
            value = newvalue.strip('\n')
        value = value.replace("\"","'")
        methods_dict[key] = value

    #ETC dictionary
    etc_dict = defaultdict()
    
    etc_contents = Tag_list.find_all(class_="table_wrap")[3].find("ul")
    for index, contents in enumerate(etc_contents.find_all("li")):
        key = re.sub(r'[\r\n\t]', '',contents.find(class_="list_tit").text).strip()
        value = contents.find(class_="list_cont").find_all(string=True)
        if len(value)==0:
            value = None
        elif len(value)==1:
            value = re.sub(r'[\r\n\t]', '',value[0].text).strip()
        else:
            newvalue=""
            for v in value:
                v = re.sub(r'[\r\n\t]', '',v.text).strip()
                newvalue += v
                if v != '':
                    newvalue += '\n'
            value = newvalue.strip('\n')
        value = value.replace("\"","'")
        etc_dict[key] = value

    #store into return_dictionary
    return_dict['url'] = url
    return_dict['main_title'] = main_title
    return_dict['short_description'] = short_description
    return_dict['summary'] = summary_dict
    return_dict['qualification'] = qualification_dict
    return_dict['methods'] = methods_dict
    return_dict['etc'] = etc_dict

    return return_dict

def insert_yp_all_overview(conn, result_list):
    cursor = conn.cursor()
    scheme_format = ['yp', 'title', 'r_number', 'url', 'main_title', 'short_description']
    for index, contents in enumerate(tqdm(result_list)):
        yp = str(contents['yp'])
        title = contents['title']
        r_number = contents['r_number']
        url = contents['contents']['url']
        main_title = contents['contents']['main_title']
        short_description = contents['contents']['short_description']
        values_format = tuple([yp, title, r_number, url, main_title, short_description])

        sql = "INSERT INTO " + "yp_all_overview " + "VALUES " + \
        "(\"{}\")".format("\", \"".join(values_format)) + \
        " ON DUPLICATE KEY UPDATE " + \
        ", ".join([f'{scheme} = "{value}"' for scheme, value in zip(scheme_format, values_format) if scheme != 'YP'])
        
        cursor.execute(sql)
        
    conn.commit()
    return

def insert_yp_summary(conn, result_list):
    cursor = conn.cursor()
    scheme_format = ['yp', 'policy_area', 'support_content', 'operation_period', 'application_period', 'supprot_scale', 'remarks']
    for index, contents in enumerate(tqdm(result_list)):
        yp = str(contents['yp'])
        policy_area = contents['contents']['summary']['정책 분야']
        support_content = contents['contents']['summary']['지원 내용']
        operation_period = contents['contents']['summary']['사업 운영 기간']
        application_period = contents['contents']['summary']['사업 신청 기간']
        supprot_scale = contents['contents']['summary']['지원 규모(명)']
        remarks = contents['contents']['summary']['비고']
        values_format = tuple([yp, policy_area, support_content, operation_period, application_period, supprot_scale, remarks])

        sql = "INSERT INTO " + "yp_summary " + "VALUES " + \
        "(\"{}\")".format("\", \"".join(values_format)) + \
        " ON DUPLICATE KEY UPDATE " + \
        ", ".join([f'{scheme} = "{value}"' for scheme, value in zip(scheme_format, values_format) if scheme != 'YP'])
        
        cursor.execute(sql)
        
    conn.commit()
    return

def insert_yp_qualification(conn, result_list):
    cursor = conn.cursor()
    scheme_format = ['yp', 'age', 'residence_income', 'education', 'major', 'employment_status', 'specialization', 'additional_info', 'eligibility']
    for index, contents in enumerate(tqdm(result_list)):
        yp = str(contents['yp'])
        age = contents['contents']['qualification']['연령']
        residence_income = contents['contents']['qualification']['거주지 및 소득']
        education = contents['contents']['qualification']['학력']
        major = contents['contents']['qualification']['전공']
        employment_status = contents['contents']['qualification']['취업 상태']
        specialization = contents['contents']['qualification']['특화 분야']
        additional_info = contents['contents']['qualification']['추가 단서 사항']
        eligibility = contents['contents']['qualification']['참여 제한 대상']
        values_format = tuple([yp, age, residence_income, education, major, employment_status, specialization, additional_info, eligibility])

        sql = "INSERT INTO " + "yp_qualification " + "VALUES " + \
        "(\"{}\")".format("\", \"".join(values_format)) + \
        " ON DUPLICATE KEY UPDATE " + \
        ", ".join([f'{scheme} = "{value}"' for scheme, value in zip(scheme_format, values_format) if scheme != 'YP'])
        
        cursor.execute(sql)
        
    conn.commit()
    return

def insert_yp_methods(conn, result_list):
    cursor = conn.cursor()
    scheme_format = ['yp', 'application_procedure', 'evaluation_announcement', 'application_website', 'required_documents']
    for index, contents in enumerate(tqdm(result_list)):
        yp = str(contents['yp'])
        application_procedure = contents['contents']['methods']['신청 절차']
        evaluation_announcement = contents['contents']['methods']['심사 및 발표']
        application_website = contents['contents']['methods']['신청 사이트']
        required_documents = contents['contents']['methods']['제출 서류']
        values_format = tuple([yp, application_procedure, evaluation_announcement, application_website, required_documents])

        sql = "INSERT INTO " + "yp_methods " + "VALUES " + \
        "(\"{}\")".format("\", \"".join(values_format)) + \
        " ON DUPLICATE KEY UPDATE " + \
        ", ".join([f'{scheme} = "{value}"' for scheme, value in zip(scheme_format, values_format) if scheme != 'YP'])
        
        cursor.execute(sql)
        
    conn.commit()
    return

def insert_yp_etc(conn, result_list):
    cursor = conn.cursor()
    scheme_format = ['yp', 'other_info', 'host_organization', 'operating_organization', 'reference_1', 'reference_2', 'attachments']
    for index, contents in enumerate(tqdm(result_list)):
        yp = str(contents['yp'])
        other_info = contents['contents']['etc']['기타 유익 정보']
        host_organization = contents['contents']['etc']['주관 기관']
        operating_organization = contents['contents']['etc']['운영 기관']
        reference_1 = contents['contents']['etc']['사업관련 참고 사이트 1']
        reference_2 = contents['contents']['etc']['사업관련 참고 사이트 2']
        attachments = contents['contents']['etc']['첨부파일']
        values_format = tuple([yp, other_info, host_organization, operating_organization, reference_1, reference_2, attachments])

        sql = "INSERT INTO " + "yp_etc " + "VALUES " + \
        "(\"{}\")".format("\", \"".join(values_format)) + \
        " ON DUPLICATE KEY UPDATE " + \
        ", ".join([f'{scheme} = "{value}"' for scheme, value in zip(scheme_format, values_format) if scheme != 'YP'])
        
        cursor.execute(sql)
        
    conn.commit()
    return

def insert_table(conn, result_list):
    insert_yp_all_overview(conn, result_list)
    insert_yp_summary(conn, result_list)
    insert_yp_qualification(conn, result_list)
    insert_yp_methods(conn, result_list)
    insert_yp_etc(conn, result_list)

def data_generation(result_list):
    with open('./env.yml') as f:
        env = yaml.load(f, Loader=yaml.FullLoader)

    #input your api
    openai.api_key = env['api_key']

    system_prompt = """You are an expert in question generation.
    Your task is, when you receive information about Korea's youth policy, 
    generate question that can deduce the answer from the policy information.

    Example : 

    예시 1:
    "지원내용: 미취업 청년 고른 취업기회 제공과 역량강화 비용 지원\n"
    Example for output :
    Q : 내가 취업을 못한 상태인데 지원받을 정책이 있을까?

    예시 2:
    "지원내용: 미취업 청년 고른 취업기회 제공과 역량강화 비용 지원\n"
    Example for output :
    Q : 현재 미취업상태 청년들에게 해당되는 혜택이 있는지 알려주세요.

    예시 3: 
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n신청 절차: 신청일 당시 도내 주민등록지에 신청\n"
    Example for output :
    Q : 어떤 절차를 거쳐야 어학자격시험 응시료를 지원 받을수 있을까?

    예시 4: 
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n신청 절차: 신청일 당시 도내 주민등록지에 신청\n"
    Example for output :
    Q : 어학자격시험 응시료를 지원받는 방법에 대해 알려주라.

    예시 5:
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n연령: 만 19세 ~ 34세\n거주지 및 소득: 도내 거주 미취업 청년(만19~34세)\n- (연령기준일) 1988. 1. 2. ~ 2004. 1. 1.(2023년 1월 1일 기준 만34세~19세)\n※ 지원 연령의 경우는 시군 조례에 따라 연령을 달리 정하고 있음\n- (요건) 응시일 기준 미취업, 지원년도 1. 1.부터 신청일까지 경기도 거주\n학력: 제한없음\n전공: 제한없음\n취업 상태: 미취업자\n특화 분야: 제한없음\n참여 제한 대상: (참여시군) 30개 시군\n※ 성남시(자체추진)\n※ 23년 1차(5월) 신청은 사업 신청일 전 예산확보 된 시군부터 개시\n"
    Example for output :
    Q : 나 30살인데 어학시험 응시료 지원 정책 받을수 있어?

    예시 6:
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n연령: 만 19세 ~ 34세\n거주지 및 소득: 도내 거주 미취업 청년(만19~34세)\n- (연령기준일) 1988. 1. 2. ~ 2004. 1. 1.(2023년 1월 1일 기준 만34세~19세)\n※ 지원 연령의 경우는 시군 조례에 따라 연령을 달리 정하고 있음\n- (요건) 응시일 기준 미취업, 지원년도 1. 1.부터 신청일까지 경기도 거주\n학력: 제한없음\n전공: 제한없음\n취업 상태: 미취업자\n특화 분야: 제한없음\n참여 제한 대상: (참여시군) 30개 시군\n※ 성남시(자체추진)\n※ 23년 1차(5월) 신청은 사업 신청일 전 예산확보 된 시군부터 개시\n"
    Example for output :
    Q : 취업을 못하고 성남시 사는데 어학시험 지원 받을수 있을까?

    예시 7:
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n연령: 만 19세 ~ 34세\n거주지 및 소득: 도내 거주 미취업 청년(만19~34세)\n- (연령기준일) 1988. 1. 2. ~ 2004. 1. 1.(2023년 1월 1일 기준 만34세~19세)\n※ 지원 연령의 경우는 시군 조례에 따라 연령을 달리 정하고 있음\n- (요건) 응시일 기준 미취업, 지원년도 1. 1.부터 신청일까지 경기도 거주\n학력: 제한없음\n전공: 제한없음\n취업 상태: 미취업자\n특화 분야: 제한없음\n참여 제한 대상: (참여시군) 30개 시군\n※ 성남시(자체추진)\n※ 23년 1차(5월) 신청은 사업 신청일 전 예산확보 된 시군부터 개시\n"
    Example for output :
    Q : 야 나 취업해야되는데 정책 괜찮은거 있어?

    예시 8:
    "제목: 미취업청년 어학자격시험 응시료 지원사업\n연령: 만 19세 ~ 34세\n거주지 및 소득: 도내 거주 미취업 청년(만19~34세)\n- (연령기준일) 1988. 1. 2. ~ 2004. 1. 1.(2023년 1월 1일 기준 만34세~19세)\n※ 지원 연령의 경우는 시군 조례에 따라 연령을 달리 정하고 있음\n- (요건) 응시일 기준 미취업, 지원년도 1. 1.부터 신청일까지 경기도 거주\n학력: 제한없음\n전공: 제한없음\n취업 상태: 미취업자\n특화 분야: 제한없음\n참여 제한 대상: (참여시군) 30개 시군\n※ 성남시(자체추진)\n※ 23년 1차(5월) 신청은 사업 신청일 전 예산확보 된 시군부터 개시\n"
    Example for output :
    Q : 제게 좋은 정책을 추천해주세요.
    """

    user_prompt = """
    Make one question using the policy information given below
    정책 정보: {passage}
    """
    
    def generate_question(passage):
        messages =[
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':user_prompt.format(passage=passage)}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response['choices'][0]['message']['content']

    for index, contents_yp in enumerate(tqdm(result_list)):
        if os.path.isfile(f'./model/train_dataset/data{index}.json'):
            continue
        try:
            #MAIN TASK : Generate Questions
            
            train_data=[]

            #TASK1 : Generate Question with 'short_description'
            content = f"지원내용: {contents_yp['contents']['short_description']}\n"

            generated_question = generate_question(content)
            train_data.append({'question':generated_question,'passage':content})

            #TASK2 : Generate Question with 'title', 'methods'
            content = f"제목: {contents_yp['contents']['main_title']}\n"
            methods = contents_yp['contents']['methods']
            for key, value in methods.items():
                if key != '신청 사이트' and value != '-' and value != None:
                    content += f'{key}: {value}\n'

            generated_question = generate_question(content)
            train_data.append({'question':generated_question,'passage':content})

            #TASK3 : Generate Question with 'title', 'qualification'
            content = f"제목: {contents_yp['contents']['main_title']}\n"
            qualification = contents_yp['contents']['qualification']
            for key, value in qualification.items():
                if value != '-' and value != None:
                    content += f'{key}: {value}\n'

            generated_question = generate_question(content)
            train_data.append({'question':generated_question,'passage':content})

            #TASK4 : Generate Question with 'title', 'summary'
            content = f"제목: {contents_yp['contents']['main_title']}\n"
            summary = contents_yp['contents']['summary']
            for key, value in summary.items():
                if value != '-' and value != None:
                    content += f'{key}: {value}\n'

            generated_question = generate_question(content)
            train_data.append({'question':generated_question,'passage':content})

            with open(f'data{index}.json', 'w', encoding='utf-8') as file:
                json.dump(train_data, file, ensure_ascii=False, indent=4)

            time.sleep(2)
            try:
                shutil.move(f'data{index}.json', "./model/train_dataset")
            except:
                shutil.copy2(f'data{index}.json', f"./model/train_dataset/data{index}.json")
                time.sleep(2)
                os.remove(f'data{index}.json')
            time.sleep(2)
        except:
            print(index)
            continue

if __name__ == '__main__':
    start_time = time.time()
    print("This code is web crawler of ChatPub Service. Final updated date is 20231120.\n")
    print("Start crawling...")

    #connect with database
    try:
        conn = mysql.connector.connect(
            user='root',
            password='1234',
            host='localhost',
            port=3306,
            database='ChatPub',
        )
    except mysql.connector.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        exit(0)
    
    #title, r_number, yp, url
    YPlist = get_id(conn)

    print(f"The number of total data : {len(YPlist)}")

    result_list = list()

    for index, contents in enumerate(tqdm(YPlist)):
        parsed_data = doc_parser(contents)
        result_list.append({
            'yp': index,
            'title': contents[0],
            'r_number': contents[1],
            'contents': parsed_data,
        })
        if index>10:
            break


    print("\nStart insertion...")
    insert_table(conn, result_list)

    # print("\nStart data generation...")
    # data_generation(result_list)

    conn.close()

    #for test:

    from model import chatbot_class
    import torch
    import faiss

    sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]

    #model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    model = chatbot_class.SentenceTransformers("klue/roberta-base")

    # MODEL_PATH 명시
    #model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # dimension
    d = 768

    # set index
    index = faiss.IndexFlatL2(d)
    Index = faiss.IndexIDMap2(index)

    # add the items into Index
    import numpy as np

    for policy in result_list:

        content = f"요약: {policy['title']}\n"
        content += f"url: {policy['contents']['url']}\n"
        content += f"제목: {policy['contents']['main_title']}\n"
        content += f"설명: {policy['contents']['short_description']}\n"
        
        for key in policy['contents']['summary']:
            value = policy['contents']['summary'][key]
            content += f'{key}: {value}\n'

        for key in policy['contents']['qualification']:
            value = policy['contents']['qualification'][key]
            content += f'{key}: {value}\n'

        for key in policy['contents']['methods']:
            value = policy['contents']['methods'][key]
            content += f'{key}: {value}\n'

        for key in policy['contents']['etc']:
            value = policy['contents']['etc'][key]
            content += f'{key}: {value}\n'

        content_vector = model.encode([content])
        vector_id = np.array([policy['yp']], dtype='int64')
        Index.add_with_ids(content_vector, vector_id)

    """    
    def search(model, index, query, k):
        query_vector = model.encode([query])
        distance, index = index.search(query_vector, k)

        # idx shape: [query 개수, k]
        # 현재는 질문이 하나이고 k=1이므로, shape: (1, 1)
        for i in index:
            # 특정 query에 대해 가장 첫 번째 결과 추출
            i = i[0]

            for key, value in result_list[i].items():
                print(f'{key}: {value}')

    #find
    #search(model, index, '문화가있는날', 1)
    """

    print(f"Process was finished. It takes {time.time()-start_time} sec.")