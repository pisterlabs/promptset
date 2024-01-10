from typing import Optional
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from .db.mongo import connect_to_mongo 
import openai
from bs4 import BeautifulSoup
import requests



class ExtractURL():
    title : str
    organization: str
    deadline : Optional[str]
    type: Optional[str]
    benefit: Optional[str]
    edulevel: Optional[str]
    major: Optional[str]
    link: Optional[str]
    requirement: Optional[str]
    def extract_info(url : str):
        mylist = load_questions()
        load_dotenv()
        key = os.getenv('KEY')
        openai.api_key = key
        openai.Model.list()
        soup = extract_url(url)
        text = soup.text
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.5,
            max_tokens = 30,
            messages = [
            {"role": "system", "content": f"Bạn là một bot đang hỗ trợ các lập trình viên để trích xuất thông tin từ một trang web, đây là dữ liệu raw text được crawl về : {text} "},
            {"role": "user", "content": f"\n{mylist[0]}"},  
        ]
        )
        response = completion.choices[0].message
        
        if response["content"].lower() == "không":
            return None
        row =[]
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.5,
            max_tokens = 200,
            messages = [
            {"role": "system", "content": f"Bạn là một bot đang hỗ trợ các lập trình viên để trích xuất thông tin từ một trang web, đây là dữ liệu raw text được crawl về : {text} "},
            {"role": "user", "content": f"(Chỉ đưa ra thông tin được hỏi, không giải thích gì thêm, mỗi câu trả lời viết trên một dòng) : \n{mylist[1]}"},
        ]
        ) 
        response = completion.choices[0].message
        row =row +response["content"].split("\n")

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.5,
            max_tokens = 750,
            messages = [
            {"role": "system", "content": f"Bạn là một bot đang hỗ trợ các lập trình viên để trích xuất thông tin từ một trang web, đây là dữ liệu raw text được crawl về : {text} "},
            {"role": "user", "content": f"(Chỉ đưa ra thông tin được hỏi,không giải thích gì thêm, mỗi câu trả lời viết trên một dòng) : \n{mylist[2]}"},
            
        ]
        )
        response = completion.choices[0].message
        row =row +response["content"].split("\n")

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.5,
            max_tokens = 200,
            messages = [
            {"role": "system", "content": f"Bạn là một bot đang hỗ trợ các lập trình viên để trích xuất thông tin từ một trang web, đây là dữ liệu raw text được crawl về : {text} "},
            {"role": "user", "content": f"(Chỉ đưa ra thông tin được hỏi, không giải thích gì thêm, mỗi câu trả lời viết trên một dòng) : \n{mylist[3]}"},
        ]
        ) 
        response = completion.choices[0].message
        row =row +response["content"].split("\n")
        
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.5,
            max_tokens = 300,
            messages = [
            {"role": "system", "content": f"Bạn là một bot đang hỗ trợ các lập trình viên để trích xuất thông tin từ một trang web, đây là dữ liệu raw text được crawl về : {text} "},
            {"role": "user", "content": f"(Chỉ đưa ra thông tin được hỏi, không giải thích gì thêm, mỗi câu trả lời viết trên một dòng) : \n{mylist[4]}"},
        ]
        ) 
        response = completion.choices[0].message
        row =row +response["content"].split("\n")
        row.append(soup.prettify())
        row.append(text)
        row.append(url)
        fields = ["title","organization","deadline","type","benefits/value","educationLevel","majors","link","requirements","html_file","raw_text","url"]
        count=0
        scholar_dict ={}
        for each in row:
            if len(each)<=1:
                continue
            info = each.split(':',maxsplit = 1)[1]
            if (count>12) :
                break
            if fields[count]=="type":
                if "hỗ trợ khó khăn" in info.lower():
                    scholar_dict[fields[count]]=1
                elif "đại học" in info.lower() or "du học" in info.lower():
                    scholar_dict[fields[count]]=2
                elif "doanh nghiệp" in info.lower() or "tổ chức" in info.lower():
                    scholar_dict[fields[count]]=3
                else:
                    scholar_dict[fields[count]]=0
            elif fields[count]=="education_level":
                level=""
                if "trung cấp" in  info.lower():
                    level= level + "1,"
                if "cao đẳng" in info.lower():
                    level = level + "2,"
                if  "đại học" in info.lower() :
                    if "sau đại học" not in info.lower():
                        level = level + "3,"
                    else: 
                        x = info.lower().strip("sau đại học")
                        if "đại học" in x:
                            level = level +"3,"
                if "thạc sĩ" in info.lower():
                    level = level + "4,"
                if "tiến sĩ" in info.lower():
                    level = level +"5"
                if len(level)==0:
                    level = level +"0"
                scholar_dict[fields[count]]=level
            elif fields[count]=="majors":
                majors =""
                if "kiến trúc" in info.lower() or "xây dựng" in info.lower():
                    majors = majors + "1,"
                if "kinh doanh" in info.lower() or "thương mại" in info.lower():
                    majors = majors + "2,"
                if "công nghệ" in info.lower() or "thông tin" in info.lower():
                    majors = majors +"3,"
                if "luật" in info.lower() or "nhân văn" in info.lower():
                    majors =majors + "4,"
                if "báo chí" in info.lower() or "khoa học xã hội" in info.lower():
                    majors = majors + "5,"
                if "y tế" in info.lower():
                    majors = majors + "6,"
                if "khoa học cơ bản" in info.lower() or "cơ bản" in info.lower():
                    majors =majors +"7,"
                if "sư phạm" in info.lower():
                    majors = majors + "8"
                if len(majors)==0:
                    majors = majors + "0"
                scholar_dict[fields[count]]=majors
            elif fields[count]=="address":
                address =""
                if "miền bắc" in info.lower():
                    address = address+"1,"
                if "miền nam" in info.lower():
                    address = address + "2,"
                if "miền trung" in info.lower():
                    address = address + "3,"
                if "châu á" in info.lower():
                    address = address + "4,"
                if "châu âu" in info.lower():
                    address = address+ "5,"
                if "mỹ" in info.lower():
                    address = address + "6,"
                if len(address)==0:
                    address = address + "0"
                scholar_dict[fields[count]]=address
            else:
                scholar_dict[fields[count]]=info
            count=count+1
        return scholar_dict