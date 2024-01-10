from db import DBhelper
import pandas as pd
import datetime
from tqdm import tqdm
import requests
from utils.log import logger
from AI_customer_service import QA_api
from langchain.output_parsers import RetryWithErrorOutputParser
from opencc import OpenCC
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from lanchain_class import title_1, title_5, sub_title
from langchain.chat_models import AzureChatOpenAI
import json
import os
import random

class Util(QA_api):
    def __init__(self):
        super().__init__('line', logger())
        self.base_web_id = 'nineyi000360'
        self.base_web_id_type = 1
        self.get_web_id()
        self.dateint = self.get_data_intdate(7)
        self.azure_openai_setting()
        self.langchain_model_setting()
        self.chat_check_model = AzureChatOpenAI(deployment_name="chat-cs-canada-4", temperature=0)
        self.article_model = AzureChatOpenAI(temperature=0.2, deployment_name='chat-cs-canada-35')
        self.article_model_16k = AzureChatOpenAI(temperature=0.2, deployment_name='chat-cs-canada-35-16k')
        self.article_4_model = AzureChatOpenAI(temperature=0.2, deployment_name='chat-cs-canada-4')

    def get_data_intdate(self, time_delay):
        return int(str(datetime.date.today() - datetime.timedelta(time_delay)).replace('-', ''))

    def get_web_id(self):
        qurey = f"""SELECT web_id,web_id_type,cx FROM missoner_web_id_table WHERE ai_article_enable = 1 """
        web_id_list = DBhelper('dione', is_ssh=True).ExecuteSelect(qurey)
        self.web_id_dict = {i: v for i, v, g in web_id_list}
        self.web_id_cx = {i: g for i, v, g in web_id_list}
        return

    def get_media_keyword_data(self):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id,article_id ,url, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > '{self.dateint}' and web_id in ("{'","'.join([i for i, v in self.web_id_dict.items() if v == 0])}")) k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article 
                WHERE `date` > {self.dateint} ) t 
        ON k.article_id = t.article_id ORDER BY t.pageviews desc;
        """
        data = DBhelper('dione', is_ssh=True).ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def get_keyword_data(self, web_id):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id, url, article_id, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > {self.dateint} and web_id = '{web_id}') k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article
                WHERE
        """
        if self.web_id_dict.get(web_id) != 1:
            query += f""" `date` > {self.dateint} and"""
        query += f""" web_id = '{web_id}') t 
                    ON k.article_id = t.article_id ORDER BY t.pageviews desc;"""
        data = DBhelper('dione', is_ssh=True).ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def langchain_model_setting(self):
        self.check_model_setting()
        self.title_model_setting()

    def check_model_setting(self):
        response_schemas = [ResponseSchema(name="check", description="判斷是否符合大眾觀賞"), ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        self.check_prompt = ChatPromptTemplate(
            messages=[SystemMessage(
                content=("你會判斷內容是否符合大眾觀賞,返回True或者False")),
                HumanMessagePromptTemplate.from_template(
                    "answer the users content as best as possible.\n{format_instructions}\n{question}")
            ],
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )

    def title_model_setting(self):
        output_parser_1 = PydanticOutputParser(pydantic_object=title_1)
        output_parser_5 = PydanticOutputParser(pydantic_object=title_5)
        output_parser_sub = PydanticOutputParser(pydantic_object=sub_title)  # 用于对输出格式化
        format_instructions_sub = output_parser_sub.get_format_instructions()
        format_instructions_1 = output_parser_1.get_format_instructions()
        format_instructions_5 = output_parser_5.get_format_instructions()
        template_1 = """
        你會根據關鍵字和關鍵字來源產生1個標題

        {keyword_info}

        標題一定要包含關鍵字,標題禁止包含任何新聞網,繁體中文回答

        -----
        {format_instructions}
        """
        template_5 = """
        你會根據關鍵字和關鍵字來源產生5個標題

        {keyword_info}

        標題一定要包含關鍵字,標題禁止包含任何新聞網,繁體中文回答

        -----
        {format_instructions}
        """
        template_sub = """
        請從主標題,生成與該主題相關的5個副標題

        主標題:{title}

        -----
        {format_instructions}
        """
        prompt_1 = PromptTemplate(  # 设置prompt模板，用于格式化输入
            template=template_1,
            input_variables=["keyword_info"],
            partial_variables={"format_instructions": format_instructions_1}
        )
        prompt_5 = PromptTemplate(  # 设置prompt模板，用于格式化输入
            template=template_5,
            input_variables=["keyword_info"],
            partial_variables={"format_instructions": format_instructions_5}
        )
        prompt_sub = PromptTemplate(  # 设置prompt模板，用于格式化输入
            template=template_sub,
            input_variables=["title"],
            partial_variables={"format_instructions": format_instructions_sub}
        )
        self.title_chain_1 = LLMChain(prompt=prompt_1, llm=AzureChatOpenAI(deployment_name="chat-cs-jp-4", temperature=0), output_parser=output_parser_1)
        self.title_chain_5 = LLMChain(prompt=prompt_5, llm=AzureChatOpenAI(deployment_name="chat-cs-jp-4", temperature=0), output_parser=output_parser_5)
        self.title_chain_sub = LLMChain(prompt=prompt_sub, llm=AzureChatOpenAI(deployment_name="chat-cs-jp-4", temperature=0), output_parser=output_parser_sub)
        self.title_chain_1 = LLMChain(prompt=prompt_1,
                                      llm=AzureChatOpenAI(deployment_name="chat-cs-canada-4", temperature=0),
                                      output_parser=output_parser_1)
        self.title_chain_5 = LLMChain(prompt=prompt_5,
                                      llm=AzureChatOpenAI(deployment_name="chat-cs-canada-4", temperature=0),
                                      output_parser=output_parser_5)
        self.title_chain_sub = LLMChain(prompt=prompt_sub,
                                        llm=AzureChatOpenAI(deployment_name="chat-cs-canada-4", temperature=0),
                                        output_parser=output_parser_sub)

    def azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')

    def translation_stw(self, text):
        cc = OpenCC('likr-s2twp')
        return cc.convert(text)

    def get_article(self, prompt, title, sub_list):
        if not sub_list:
            response_schemas = [ResponseSchema(name=f"Articles", description=f"Articles")]
            model = self.article_4_model
        else:
            response_schemas = [ResponseSchema(name=f"paragraph_{i + 1}", description=f"Articles with subtitle '{v}'")
                                for i, v in enumerate(sub_list)]
            model = self.article_model
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        _input = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template("{question}\n{format_instructions}\n")],
            input_variables=["question"], partial_variables={"format_instructions": format_instructions}).format_prompt(
            question=prompt)
        output = model(_input.to_messages())
        # noinspection PyBroadException
        try:
            gpt_res = output_parser.parse(output.content)
        except:
            print('產生失敗！！,重新產生')
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=self.article_model_16k)
            gpt_res = retry_parser.parse_with_prompt(output.content, _input)
        if not sub_list:
            res = [
                gpt_res['Articles'].replace('文章標題：', '').replace('文章標題:', '').replace('文章內容：', '').replace(
                    '文章內容:', '')]
        else:
            res = [v.replace(f'{i}。', '').replace(f'{i}', '') for i, v in zip(sub_list, gpt_res.values())]
        return res

    def get_generate_articles_prompt(self, title, sub_list, keyword_info_dict, ta_setting):
        if not ta_setting:
            print(f"產生無TA文章,段落：{len(sub_list)}")
            style = 'easy'
        else:
            print(f"產生有TA文章,段落：{len(sub_list)}")
            gender, age, income, interests, occupation, style = ta_setting
        if not sub_list:
            prompt = f"""Please use the following provided information to craft the concise article of approximately 100 words,including the title, subtitles, keywords, keyword explanations,Target Audienceand,and other relevant details. The article should be written in an {style} style and in Taiwan Mandarin. Ensure that proper grammar and sentence structure are maintained throughout.

            Title: "{title}"
            """
        else:
            prompt = f"""Please use the following provided information to craft concise article of approximately {str(len(sub_list))}00 words, including the title, subtitles, keywords, keyword explanations,Target Audience, and other relevant details. The article should be written in an {style} style and in Taiwan Mandarin. Ensure that proper grammar and sentence structure are maintained throughout.Please ensure that the subtitles are not altered in any way.

            Title: "{title}"

            Subtitles:
            """
            prompt += '     '.join([f'{str(i + 1)}. "{v}"\n' for i, v in enumerate(sub_list)])
        prompt += """
        Keywords and Explanations:
        """
        prompt += '     '.join(
            [f"""{i + 1}."{data[0]}": "{data[1][0]}"\n""" for i, data in enumerate(keyword_info_dict.items())])
        if ta_setting:
            prompt += f"""
            Target Audience information and Interests:
            - Gender: "{gender}"
            - Age: "{age}"
            - Income bracket: "{income}"
            - Interests: "{interests}"
            - Occupation: "{occupation}"
            """
        else:
            prompt = prompt.replace(",TargetAudience", '')
        prompt += "Please focus on incorporating each keyword in the article in a manner that reflects a clear understanding of its meaning while considering the target audience provided. Be creative, and ensure that each subtitle has its own section in the article.Please ensure that the subtitles are not altered in any way. "
        if not ta_setting:
            prompt = prompt.replace(" while considering the target audience provided", '')
        if not sub_list:
            prompt += "and there should be no explanations within the responses only response article.don't add any subtitles"
        print(f"""PROMPT:{prompt}""")
        return prompt


class AiTraffic(Util):
    def __init__(self):
        super().__init__()
        self.media_keyword_pd = self.get_media_keyword_data()
        self.keyword_all_set = set(self.media_keyword_pd.keyword)
        self.all_keyword_pd = {}
        self.get_keyword_pd()

    def get_keyword_info(self, web_id, keyword):
        print(f"""獲取{web_id}的關鍵字資訊""")
        if web_id not in self.web_id_dict or web_id == 'pure17':
            print(f"""不包含{web_id}的關鍵字資訊,更改為base_web_id:{self.base_web_id}""")
            web_id = self.base_web_id
        keyword_list = keyword.split(',')
        keyword_list = [i for i in keyword_list if i]
        cx = self.web_id_cx[web_id]
        df = self.all_keyword_pd[web_id]
        keyword_info_dict = {}
        visited = []
        print(keyword_list)
        for key in keyword_list:
            if len(df) and key in set(df.keyword):
                curr_keyword_info = df[df.keyword == key]
                for title, content, article_id, img in curr_keyword_info[['title', 'content', 'url', 'image']].values:
                    if self.check_news(title):
                        keyword_info_dict[key] = (title, content, web_id, article_id, img)
                        print(f'內站有關鍵字：{key}')
                        break
            elif key in self.keyword_all_set:
                curr_keyword_info = self.media_keyword_pd[self.media_keyword_pd.keyword == key]
                for title, content, web_id_article, article_id, img in curr_keyword_info[
                    ['title', 'content', 'web_id', 'url', 'image']].values:
                    if self.check_news(title):
                        keyword_info_dict[key] = (title, content, web_id_article, article_id, img)
                        print(f'外站有關鍵字：{key}')
                        break
            else:
                if len(keyword_list) > 2:
                    t = 0
                    while True:
                        if t == 10:
                            search_key = key
                            break
                        r = random.choice(keyword_list)
                        if set([key, r]) in visited or r == key:
                            t += 1
                            continue
                        visited.append(set([key, r]))
                        search_key = f'{key}+{r}'
                        break
                else:
                    search_key = key

                html1 = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx={cx}&key={self.Search.GOOGLE_SEARCH_KEY}&q={key}'
                html2 = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx=41d4033f0c2f04bb8&key={self.Search.GOOGLE_SEARCH_KEY}&q={search_key}'
                for i, html in enumerate([html1, html2]):
                    r = requests.get(html, timeout=5)
                    if r.status_code != 200:
                        continue
                    res = r.json().get('items')
                    if not res:
                        continue
                    for data in res:
                        title = data.get('htmlTitle')
                        sn = data.get('snippet')
                        link = data.get('link')
                        if self.check_news(title + sn):
                            keyword_info_dict[key] = (title, sn, 'google', link, '_')
                            break
                    if key in keyword_info_dict:
                        if i == 0:
                            print(f'google搜尋主要關鍵字{key}')
                        if i == 1:
                            print(f'google搜尋主要關鍵字:{key},組合關鍵字:{search_key}')
                        break
                if key not in keyword_info_dict:
                    print(f'無關鍵字：{key}資訊')
                    keyword_info_dict[key] = ('_', '_', 'none', '_', '_')

        print(f"""關鍵字資訊:{keyword_info_dict}""")
        return keyword_info_dict

    def get_keyword_pd(self):
        pbar = tqdm(list(self.web_id_dict.keys()))
        for i, web_id in enumerate(pbar):
            pbar.set_description(web_id)
            self.all_keyword_pd[web_id] = self.get_keyword_data(web_id)

    def check_news(self, text):
        _input = self.check_prompt.format_prompt(question=text)
        output = self.chat_check_model(_input.to_messages())
        if 'False' in output.content:
            return False
        return True

    def check_keyword(self, keywords, web_id):
        print('檢查關鍵字是否包含')
        if web_id not in self.web_id_dict:
            web_id = self.base_web_id
        df = self.all_keyword_pd[web_id]
        keywords_set = set(df.keyword)
        keyword_list = keywords.split(',')
        for key in keyword_list:
            if key in self.keyword_all_set or key in keywords_set:
                print(f'關鍵字：{key}')
                return True
        return False


    def get_title(self, web_id: str = 'test', user_id: str = '', keywords: str = '', web_id_main: str = '', article: str = None, types: int = 1):
        print(f"""輸入web_id:{web_id}""")
        # check_keyword = self.check_keyword(keywords, web_id_main) if web_id_main else self.check_keyword(keywords, web_id)
        # if not check_keyword:
        #     pass
        keyword_info_dict = self.get_keyword_info(web_id_main, keywords) if web_id_main else self.get_keyword_info(
            web_id, keywords)
        prompt = ''.join([f"關鍵字:{i}\n'{i}'關鍵字的來源:{v[0]}\n\n" for i, v in keyword_info_dict.items()])
        if types == 1:
            result = self.title_chain_1.run({'keyword_info': prompt})
            title = self.translation_stw(result.title)
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords, title, json.dumps([{'keyword': keyword, 'title': data[0], 'web_id': data[2], 'url': data[3], 'image': data[4]} for keyword, data in keyword_info_dict.items()]), datetime.datetime.now()]],columns=['user_id', 'web_id', 'type', 'inputs', 'title', 'keyword_dict', 'add_time']), db='sunscribe',table='ai_article', chunk_size=100000, is_ssh=False)
            return [title]
        else:
            if not article:
                return {"message": "no article input"}
            result = self.title_chain_5.run({'keyword_info': prompt})
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords, self.translation_stw(result.title[0]),self.translation_stw(result.title[1]), self.translation_stw(result.title[2]), self.translation_stw(result.title[3]), self.translation_stw(result.title[4]),article, 2]],
                                                       columns=['user_id', 'web_id', 'type', 'inputs', 'subheading_1','subheading_2', 'subheading_3', 'subheading_4', 'subheading_5', 'article_1', 'article_step']), db='sunscribe', table='ai_article', chunk_size=100000, is_ssh=False)
            return result.title

    def get_sub_title(self, title: str = '', user_id: str = '', web_id: str = 'test', types: int = 1):
        print(f"""獲取副標題,標題為:{title}""")
        result = self.title_chain_sub.run({'title': title})
        print(f"""副標題產生成功,副標題為:{result.sub_title}""")
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, title, result.sub_title[0],
                                                     result.sub_title[1], result.sub_title[2], result.sub_title[3],
                                                     result.sub_title[4]]],
                                                   columns=['user_id', 'web_id', 'type', 'title', 'subheading_1',
                                                            'subheading_2', 'subheading_3', 'subheading_4',
                                                            'subheading_5']), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return {i + 1: v for i, v in enumerate(result.sub_title)}

    def generate_articles(self, title: str = '', subtitle_list: list = [], keywords: str = '', user_id: str = '',
                                 web_id: str = 'test', types: int = 1, ta: list = []):
        query = f"SELECT keyword_dict  FROM web_push.ai_article WHERE web_id = '{web_id}' and user_id  ='{user_id}'"
        keyword_info_db = DBhelper('sunscribe').ExecuteSelect(query)
        sub_list = [i for i in subtitle_list if i]
        if keyword_info_db:
            print('db有keyword_info')
            keyword_info_dict = {i['keyword']: (i['title'], i['web_id'], i['url']) for i in
                                 json.loads(keyword_info_db[0][0])}
        else:
            print('db無keyword_info,可能有問題')
            keyword_info_dict = self.get_keyword_info(web_id, keywords)
        prompt = self.get_generate_articles_prompt(title, sub_list, keyword_info_dict, ta)
        res = self.get_article(prompt, title, sub_list)
        print(f"""產生的文章內容:\n{res}""")
        return res


if __name__ == " __main__":
    AI_traffic = AiTraffic()
