import requests
import bs4
import pandas as pd
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
#from config import model_engine,template_for_structured,reference_txt_path,template_final_answer,message_txt_path
from . import config
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import re
from langchain.document_loaders.csv_loader import CSVLoader
import openai
import csv
##構造化出力(JSON) パーサー用のimport
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import sys

def search_google(keywords_list):
    serach_num=5
    #keyrords_listの各要素を一行のテキスト化してsearch_keywordに格納
    search_keyword = ''
    for keyword in keywords_list:
        search_keyword += keyword
    print('search_key'+search_keyword)
    #search_keywordをgoogle検索(上位5件)
    print('start1')
    search_url = f'https://www.google.co.jp/search?hl=ja&num={serach_num}&q=' + search_keyword + '電話番号'
    print('search_url'+search_url)
    res_google = requests.get(search_url)
    print(res_google)
    print('start2')
    #res_google.textは、検索結果のページのHTML
    bs4_google = bs4.BeautifulSoup(res_google.text, 'lxml')
    google_search_page = bs4_google.select('div.kCrYT>a')
    print(google_search_page)
    
    #rank:検索順位
    rank = 1
    site_rank = []
    site_title = []
    site_url = []
    #!seleniumURLloaderよりbsのget_text()の方が早い？？
    site_content = []
    print('start3')
    for site in google_search_page:
        try:    
            site_title.append(site.select('h3.zBAuLc')[0].text)
            url = site.get('href').split('&sa=U&')[0].replace('/url?q=', '')
            site_url.append(url)
            site_rank.append(rank)
            rank +=1
            #取得したURLの全てのテキスト要素を取得する
            res_google_content = requests.get(url)
            bs4_google_content = bs4.BeautifulSoup(res_google_content.text, 'lxml')
            text_content=bs4_google_content.get_text()
            #正規表現で任意の連続する/nやらタブ、スペースを単一のスペースに変換(文字数削減)
            re_text_content = re.sub("\s+"," ",text_content)
            print(len(re_text_content))
            site_content.append(re_text_content)
        except IndexError:
            continue
    print('start4')
    
    print('start5')
    #!contentを要約（だいたい16kにおさるのでそのまま掘り込むが、よくないよ。あとモジュール化しろ）
    site_content_summarized = []
    skip_number_list=[]
    for i in range(len(site_content)):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-1106',
                messages=[
                    {
                        "role": "system",
                        "content": "以下のテキストを要約し、最も重要なポイントを箇条書きにまとめてください。ただし、電話番号や住所がある場合はそれらを含めてください。"
                    },
                    {
                        "role": "user",
                        "content": site_content[i]
                    }
                ],
                )
            message = response["choices"][0]["message"]["content"]
            #print(message)
            print(len(message))
            site_content_summarized.append(message)
        except:
            print('token制限のためスキップ')
            #何番目がスキップされたのか記憶
            skip_number_list.append(i)
            continue
    #skip_number_listが空でない時、スキップされた要素を削除
    if len(skip_number_list) != 0:
        # インデックスを大きい順にソート(要素数が大きい方から消さないとこの方法だとズレるよ)
        skip_number_list.sort(reverse=True)
        for index in skip_number_list:
            #.popで各リストを削除
            site_rank.pop(index)
            site_title.pop(index)
            site_url.pop(index)
            site_content.pop(index)
    
    df = pd.DataFrame({'rank':site_rank, 'title':site_title, 'URL':site_url,'summarized_content':site_content_summarized ,'content':site_content})
    df_summary = pd.DataFrame({'rank':site_rank, 'title':site_title, 'URL':site_url,'summarized_content':site_content_summarized})
    # df.to_csv('tmp/csv/search_google_result.csv', index=False)
    # df_summary.to_csv('tmp/csv/search_google_result_summary.csv', index=False)
    #こっからlc使用
    #urlをlcloaderに入力
    
    #return [site_rank,site_title,site_url,site_content_summarized,site_content]
    print(df_summary)
    return [site_rank,site_title,site_url,site_content_summarized]
    #!SeleniumURLloaderが遅い！！！！
    # loader = UnstructuredURLLoader(urls=site_url)
    # datas = loader.load()
    # # # print(datas)
    # # print('OK')
    # # print(type(datas))
    # # print(datas[0])
    # # print(type(datas[0]))
    # #!csvloaderを使おう
    # summary_list = []
    # llm = OpenAI(model_name=model_engine,temperature=0)
    # text_splitter = CharacterTextSplitter()
    # for data in datas:
    #     # summary = text_splitter.split_text(data)
    #     # docs = [Document(page_content=t) for t in summary[:3]]
    #     chain = load_summarize_chain(llm, chain_type="map_reduce")
    #     result = chain.run(data)
    #     summary_list.append(result)
    # print(summary_list)
    
def get_Townpage_by_google(keywords_list):
    #keyword.textファイルを読み込んでkeywords_listに代入
    #初期化
    #?-------------------------------
    # keywords_list = []
    # with open('tmp/text/keywords_for_google.txt', "r", encoding="utf-8") as f:
    #     for keyword in f:
    #         keywords_list.append(keyword.rstrip('\n'))
    result_google_seach = search_google(keywords_list)
    print(result_google_seach,len(result_google_seach),len(result_google_seach[0]))
    #?-------------------------------
    #return_text = convert_csv_to_text('tmp/csv/search_google_result_summary.csv')
    # #保存
    # with open(config.reference_txt_path, "w") as f:
    #     f.write(return_text)
    
    response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106',
            messages=[
                {
                    "role": "user",
                    "content": f"""
INPUT:以下の参考文をもとに候補を複数個(1~3個程度)箇条書きで挙げてください
■制約条件①電話番号が重複した際はrankの低いものを削除してください。
■制約条件②箇条書きの内容は
・店名or場所の名称
・電話番号
・簡単な紹介文(数十文字程度)
の三つとしてください

■参考文
{result_google_seach}
OUTPUT:
"""
                }
            ],
            temperature=0.0
            )
    message = response["choices"][0]["message"]["content"]
    # with open(config.message_txt_path ,"w") as f:
    #     f.write(message)
    # model = ChatOpenAI(model_name='text-davinci-003', temperature=0.0)
    # prompt_final_answer = PromptTemplate(
    #     template=template_final_answer,
    #     input_variables=["input"]
    # )
    # _input = prompt_final_answer.format_prompt(input=return_text)
    # output = model(_input)
    # print(output)
    return message

    #?折角なのでjson形式で出力
    #model定義 賢いの使った方が良さげ(token制限から普通にgpt35でやってます)
#     model_name_pydantic = model_engine
#     temperature = 0.0
#     #構造化定義
#     response_schemas = [
#     ResponseSchema(name="result_num", description="最終的な候補の数"),
#     ResponseSchema(name="rank", description="検索順位"),
#     ResponseSchema(name="tel", description="電話番号の候補"),
#     ResponseSchema(name="tel", description="それぞれの候補に対する店名や場所の名称"),
#     ResponseSchema(name="source", description="それぞれの候補の参考URL"),
# ]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
#     format_instructions = output_parser.get_format_instructions()
#     prompt_structured_parser = PromptTemplate(
#     template=template_for_structured,
#     input_variables=["reference"],
#     partial_variables={"format_instructions": format_instructions}
# )
#     model = OpenAI(model_name=model_name_pydantic, temperature=temperature)
#     #参考文をリストから読みやすい形に変換
#     reference_text = generate_reference_text(result_google_seach[:5])
#     print(reference_text)
#     _input = prompt_structured_parser.format_prompt(reference=reference_text)
#     output = model(_input.to_string())
#     print(output)
#     final_result=output_parser.parse(output)
#     print(final_result)