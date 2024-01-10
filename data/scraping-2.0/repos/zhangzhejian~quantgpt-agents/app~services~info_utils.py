import sys
sys.path.append('../')
import qstock as qs
from typing import List, Optional
import time
import asyncio
from langchain.llms import OpenAI

keywords_response_format = {"keywords":["keyword","keyword"]}
def get_gpt_completion(prompt):
    llm = OpenAI(temperature=0.9)
    resp = llm.generate([prompt])
    print(resp.generations[0][0].text)
    return resp.generations[0][0].text

class InformationManager(object):
    def __init__(self, stock_code):
        self.stock_code = stock_code

    def get_stock_df(sekf,code):
        return qs.get_data(code)

    def search_main_business(self, stock):
        df=qs.main_business(stock)
        select_period = df.head(1)['报告期'].values[0]
        cols = ['分类','营业收入(万)', '占主营收入比', '毛利率']
        df = df[(df['报告期'] == select_period) & (df['分类方向'] == '按行业分')][cols]
        dic = df.to_dict(orient='records')
        if len(dic) <= 0:
            return None
        result = {}
        result[f'报告期:{select_period}'] = dic
        return result

    def get_all_industry_name_list() -> List[str]:
        name_list=qs.ths_index_name('行业')
        return name_list

    '''
    通过GPT总结新闻标题和内容的行业关键词
    用于后续匹配股票的主营业务、所属行业、概念
    '''
    def determine_news_keywords(self):
        news = qs.news_data()
        for index, row in news.iterrows():
            if index >= 1:
                return
            prompt = self.generate_prompt_news_keywords(title= row['标题'], content= row['内容'])
            print(prompt)
            print(get_gpt_completion(prompt))

    def generate_prompt_news_keywords(self, title: str, content: str) -> str:
        return (
            "Role Description:\nYou are a profession financial analyzer. You are responsible for detecting all news' effect on all industries. You should consider what the industries will be affected by the news directly or potentially.\n\n"
            "##Constraints:\n"
            "1. You should only respond in JSON format as described below: \n"
            f"  Response Format: ```{keywords_response_format} ```"
            "Ensure the response can be parsed by Python json.loads\n"
            "2. Just give the answer, don't explain reasons\n"
            f"\nnews title:{title}\n"
            f"content: {content}\n"
            "Begin.Please give all the keywords related to industries. \n"
            "Result:"
        )


news_manager = NewsManager("")
def determine_news_correlation(title, content, stock_information):
    return

if __name__ == '__main__':
    # print(get_all_industry_name_list())
    # determine_news_keywords()
    # search_main_business('000155')
    # name_list=qs.ths_index_name('行业')
    # print(len(name_list))
    # # print(df=qs.eps_forecast())
    # df=qs.intraday_money('中国平安')
    # print(df)
    # print(qs.stock_snapshot('中国平安'))
    news = qs.news_data()
    news.to_csv('news.csv')
    df=qs.get_data('000155')
    df = df.tail(20)
    # print(df)
    print(df['close'].values.tolist())
    print(df['volume'].values.tolist())

    df=qs.get_data('000155',start='20220928',freq=101)
    print(df)
    df = df.tail(80)
    # print(df)
    print(df['close'].values.tolist())
    print(df['volume'].values.tolist())
    # df=qs.main_business('000155')
    print(news_manager.search_main_business('000155'))