import os
import time

from AI_customer_service import QA_api
from utils.log import logger
from db import DBhelper
import pandas as pd
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import itertools
import openai
import re


class AI_Search(QA_api):
    def __init__(self):
        super().__init__('line',logger())
        self.get_web_id_list()
        self.all_subdomain_dict = self.get_sub_domain_dict()
        self.all_product_rank_dict = self.get_product_rank_dict()
        self.Azure_openai_setting()
        self.get_langchain_setting()

    def get_web_id_list(self):
        self.web_id_list = []
        for web_id,data in self.CONFIG.items():
            if data['ai_Search']:
                self.web_id_list.append(web_id)
    def get_sub_domain_dict(self):
        if 'nineyi000360' in self.web_id_list:
            self.web_id_list.append('famimarketing')
        query = f"""SELECT web_id,subdomain  FROM web_push.all_website_category x WHERE web_id in ('{"','".join(self.web_id_list)}')"""
        Data = DBhelper('sunscribe').ExecuteSelect(query)
        sub_domain_dict = {web_id: subdomain for web_id, subdomain in Data}
        if 'famimarketing' in sub_domain_dict:
            sub_domain_dict['nineyi000360'] = sub_domain_dict['famimarketing']
        return sub_domain_dict

    def get_product_rank_dict(self):
        query = f"""SELECT web_id,product_id,rank FROM web_push.all_hot_items x WHERE web_id in ('{"','".join(self.web_id_list)}')"""
        Data = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query)
        rank_dict = {i: {} for i in self.web_id_list}
        for web_id, product_id, rank in Data:
            if web_id in ['familyapp','famimarketing']:
                web_id = 'nineyi000360'
            if product_id in rank_dict[web_id]:
                continue
            rank_dict[web_id][product_id] = rank
        return rank_dict

    def get_subdomain_url(self,web_id, product_id):
        sub_domain = self.all_subdomain_dict.get(web_id)
        if not sub_domain:
            return '_'
        return sub_domain + '/avivid/product/detail/' + product_id

    def get_rank(self,web_id, product_id):
        rank_dict = self.all_product_rank_dict.get(web_id)
        if not rank_dict:
            return 999
        rank = rank_dict.get(product_id)
        if not rank:
            return 999
        return rank

    def get_Fuzzy_keyword(self,keyword):
        keyword_list = list(keyword)
        keyword_combination = []
        for i in range(len(keyword_list), 0, -1):
            for j in itertools.combinations(keyword_list, i):
                keyword_combination.append(j)
        return [''.join(i) for i in keyword_combination]
    def get_hot_product_info(self,web_id):
        query = f"""SELECT s.web_id,s.product_id,s.title,s.url,s.image_url,k.rank From 
        (SELECT product_id,rank FROM web_push.all_hot_items WHERE web_id = '{web_id}' order by rank limit 10 ) as k INNER join (SELECT web_id,product_id ,title ,url,image_url FROM web_push.item_list x WHERE web_id = '{web_id}') as s 
        on k.product_id = s.product_id"""
        df = pd.DataFrame(DBhelper('rhea1-db0', is_ssh=True).ExecuteSelect(query)).sample(3).reset_index()
        json = {}
        for i, data in df.iterrows():
            json[i] = {'title': data['title'], 'url': data['url'], 'img_url': data['image_url']}
        return json
    def get_similar_info(self,web_id,product_id_list):
        for i in product_id_list:
            query = f"""SELECT s.title,s.url,s.image_url From 
                        (SELECT DISTINCT similarity_product_id FROM web_push.item_similarity_table x WHERE web_id ='{web_id}' and  main_product_id ='{i}' and similarity_product_id not in ('{"','".join(product_id_list)}') order by rank limit 3) as k 
                    INNER join (SELECT web_id,product_id ,title ,url,image_url FROM web_push.item_list x WHERE web_id = '{web_id}') as s
                    on k.similarity_product_id = s.product_id
                    """
            df = pd.DataFrame(DBhelper('rhea1-db0', is_ssh=True).ExecuteSelect(query))
            if len(df) != 0:
                break
        json = {}
        for i, data in df.iterrows():
            json[i] = {'title': data['title'], 'url': data['url'], 'img_url': data['image_url']}
        return json



    def get_product_info(self,web_id, keyword_info,n=3):
        keyword = keyword_info.get('keyword')
        Fuzzy_keyword = self.get_Fuzzy_keyword(keyword)
        price = keyword_info.get('price').replace('元', '').replace('塊', '')
        price_range = keyword_info.get('price_range').replace('元', '').replace('塊', '')
        price_sorting = keyword_info.get('price_sorting')
        for k in Fuzzy_keyword:
            query = f"""SELECT k.keyword,s.web_id,s.product_id,s.title,s.description,s.price,s.image_url From
            (SELECT article_id,keyword FROM web_push.keyword_article_list_no_utf8 x WHERE web_id = '{web_id}' and keyword = '{k}') as k INNER join (SELECT web_id,product_id ,title ,description ,price,image_url FROM web_push.item_list x WHERE web_id = '{web_id}') as s 
            on k.article_id = s.product_id"""
            if web_id == 'nineyi000360':
                query = query.replace("web_id = 'nineyi000360'","""web_id in ("nineyi000360","familyapp","famimarketing")""")
            Data = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query)
            df = pd.DataFrame(Data)
            if len(df) != 0:
                break
        if len(df) == 0:
            return df
        df['sub_url'] = df.apply(lambda x: self.get_subdomain_url(x.web_id if x.web_id  not in ['familyapp','famimarketing'] else 'nineyi000360' ,x.product_id), axis=1)
        df['rank'] = df.apply(lambda x: self.get_rank(x.web_id if x.web_id  not in ['familyapp','famimarketing'] else 'nineyi000360', x.product_id), axis=1)
        if price_sorting and price_sorting != 'None':
            if price_sorting == 'True':
                df = df.sort_values('price').reset_index(drop='index')
            else:
                df = df.sort_values('price', ascending=False).reset_index(drop='index')
        df = df.sort_values('rank').reset_index(drop='index')
        if price_range != 'False':
            if price_range.endswith('-'):
                pri = price_range.split('-')[0]
                df = df[(df['price'] >= int(pri))]
                df = df.sort_values('price',ascending=True).reset_index(drop='index')
            elif price_range.startswith('-'):
                pri = price_range.split('-')[0]
                df = df[(df['price'] <= int(pri))]
                df = df.sort_values('price')
            elif '-' in price_range:
                pri = price_range.split('-')
                df = df[(df['price']>=int(pri[0])) & (df['price'] <= int(pri[1]))]
                df = df.sort_values('price',ascending=True).reset_index(drop='index')
        elif price != 'False':
            df = df[(df['price'] <= int(price))]
            df = df.sort_values('price',ascending=False).reset_index(drop='index')
        return df.iloc[:n]
    def get_gpt_query_serch(self,df , message: str, web_id_conf: dict,web_id):
        '''
        :param query: result from likr_search
        :param query: question for chatgpt
        -------
        chatgpt_query
            Results:

            [1] "result[0]['htmlTitle']}",snippet= "{result[0]['snippet']}",description = "{result[0]['pagemap']['metatags'][0]['og:description']"

            [2] "result[1]['htmlTitle']}",snippet= "{result[1]['snippet']}",description = "{result[1]['pagemap']['metatags'][0]['og:description']"

            [3] "result[2]['htmlTitle']}",snippet= "{result[2]['snippet']}",description = "{result[2]['pagemap']['metatags'][0]['og:description']"

            Current date: {date}

            Instructions: If you are "{web_id_conf['web_name']}" customer service. Using the information of results or following the flow of conversation, write a comprehensive reply to the given query in 繁體中文 and following the rules below:
            Always cite the information from the provided results.
            "親愛的顧客您好，" in the beginning.
            "祝您愉快！" in the end.
            Query: {message}
        '''
        gpt_query = [{"role": "system", "content": f"我們是{web_id_conf['web_name']}(代號：{web_id_conf['web_id']},網站：{self.all_subdomain_dict.get(web_id)}),{web_id_conf['description']}"}]
        if type(df) != str:
            chatgpt_query = f"""You are a GPT-4 customer service robot for "{web_id_conf['web_name']}". Your task is to respond to customer inquiries in 繁體中文. Always start with "親愛的顧客您好，" and end with "祝您愉快！". Your objective is to provide useful, accurate and concise information that will help the customer with their concern or question. You have to use information from the information provided,Do not generating content that is not directly related to the customer's questions or any information.\n Information:"""
            for i, v in df.iterrows():
                if v.get('title'):
                    chatgpt_query += f"""\n\n[{i + 1}] "{v.get('title')}"""
                if v.get('description'):
                    chatgpt_query += f""",description = "{v.get('description')}"""
                if v.get('price'):
                    chatgpt_query += f""",price = "{v.get('price')}"""
            chatgpt_query += f"""\n\nCustomer question:{message}"""
        else:
            chatgpt_query = f"""Act as customer service representative for "{web_id_conf['web_name']}"({web_id_conf['web_id']}). Provide a detailed response addressing their concern, but there is no information about the customer's question in the database.  Reply in 繁體中文 and Following the rule below:\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {message}"""
        #####################################################################################
        gpt_query += [{'role': 'user', 'content': chatgpt_query}]
#         while self.num_tokens_from_messages(gpt_query) > 3500 and len(gpt_query) > 3:
#             gpt_query = [gpt_query[0]] + gpt_query[3:]
        return gpt_query

    def get_keyword_info(self,message):
        _input = self.prompt.format_prompt(question = message)
        output = self.chat_model(_input.to_messages())
        return self.output_parser.parse(output.content)
    def get_langchain_setting(self):
        response_schemas = [ResponseSchema(name="keyword", description="Most important product titles or keywords"),
                            ResponseSchema(name="price",description='If the question ask for price,return the price else return "False"'),
                            ResponseSchema(name="price_range",description='If the question ask for price range,return the price range which sep by "-" else return "False"'),
                            ResponseSchema(name="price_sorting",description='If you ask for the cheapest return "True", if you ask for the most expensive return "False". else return "None"')]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template="You are a robot that determines the products and prices of customer problems.'元' and '塊' means $\n{format_instructions}\n{question}",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )
        self.chat_model = AzureChatOpenAI(deployment_name="chat-cs-jp-4",temperature=0)
    def main(self,web_id,message):
        print(f'客戶輸入:{message}')
        keyword_info = self.get_keyword_info(message)
        print(keyword_info)
        prodcut_info = self.get_product_info(web_id,keyword_info)
        product_json = self.get_product_json(prodcut_info)
        print(product_json)
        if not product_json:
            ans = '很抱歉,目前查無商品,請更改價格區間或者更換商品搜尋'
        else:
            query = self.get_gpt_query_serch(prodcut_info,message,self.CONFIG[web_id],web_id)
            gpt_answer = self.ChatGPT.ask_gpt(query)
            ans = self.adjust_ans_format(gpt_answer)
            ans = re.sub('\w*(抱歉|對不起|我們沒有)\w*(，|。)','',ans)
        hot_product_json = self.get_hot_product_info(web_id)
        return {'res': ans, 'product': product_json, 'hot': hot_product_json}
    def main_sim(self,web_id,message):
        print(f'客戶輸入:{message}')
        keyword_info = self.get_keyword_info(message)
        print(keyword_info)
        prodcut_info = self.get_product_info(web_id,keyword_info)
        product_json = self.get_product_json(prodcut_info)
        print(product_json)
        if not product_json:
            ans = '很抱歉,目前查無商品,請更改價格區間或者更換商品搜尋'
        else:
            query = self.get_gpt_query_serch(prodcut_info,message,self.CONFIG[web_id],web_id)
            gpt_answer = self.ChatGPT.ask_gpt(query)
            ans = self.adjust_ans_format(gpt_answer)
            ans = re.sub('\w*(抱歉|對不起|我們沒有)\w*(，|。)','',ans)
        sim_product_json = self.get_similar_info(web_id,list(prodcut_info['product_id']))
        return {'res': ans, 'product': product_json, 'hot': sim_product_json}


    def get_product_json(self,df):
        json = {}
        for i,data in df.iterrows():
            json[i] = {'title':data['title'],'price':int(data['price']),'img_url':data['image_url'],'url':data['sub_url']}
        return json

    def Azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')


if __name__ == "__main__":
    Search = AI_Search()
    start = time.time()
    print(Search.main('nineyi000360', '最貴的馬克杯'))
    end = time.time()
    print(f'熱銷品執行時間{end-start}')
    #print(Search.main_sim('nineyi000360', '衛生紙'))
    end2 = time.time()
    print(f'類品執行時間{end2 - end}')