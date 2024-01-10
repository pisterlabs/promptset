# 关系抽取
import json
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from units.merge_json import merge_json
from tqdm import tqdm
from units.load_data import load_data


async def relation_extraction(pages):
    model = "Qwen-14B-Chat-Int4"

    examples = [
        {"input": "雅生活服务（03319.HK）成立于1992年，是开发商雅居乐集团的附属公司。2017年公司收购绿地物业并引入绿地控股（600606.SH）成为其长期战略股东。2018 年 2月公司从雅居乐集团拆分后在港交所上市。公司业务涉及住宅物业服务、高端商写资产管理、公共物业服务、社区商业。按照公司收入来源划分，主营业务分为物业管理服务。",
         "output": '''{"relation_list":[{"h_name":"雅生活服务", "relation":"隶属","t_name" ="雅居乐集团"}, {"h_name":"雅生活服务", "relation":"并购","t_name" ="绿地物业"},{"h_name":"雅生活服务", "relation":"开展","t_name" ="物业管理服务"},{"h_name":"绿地控股", "relation":"投资","t_name" ="雅生活服务"}]}'''},
        {"input": '''发展规划引导行业稳步发展：与此同时，根据工信部《新能源汽车产业发展规划（2021-2025）》（征求意见稿），到2025年，新能源汽车销量占当年汽车总销量的25%，按照2018年汽车销量（约2800万辆）进行计算，2025年新能源汽车销量将达到700万辆，对应2018-2025年CAGR为28%；并专列“保障措施”章节，地方政府加大公共车辆运营、21年重点区域公共领域新增车辆全部电动化。''',
         "output": '''{"relation_list":[{"h_name":"工信部", "relation":"发布","t_name" ="《新能源汽车产业发展规划（2021-2025）》（征求意见稿）"}]}'''}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    relation_list = ['{{"头实体": "行业","关系类型": "隶属","尾实体": "行业"}}',
                     '{{"头实体": "机构","关系类型": "属于","尾实体": "行业"}}',
                     '{{"头实体": "行业","关系类型": "拥有","尾实体": "风险"}}',
                     '{{"头实体": "机构","关系类型": "拥有","尾实体": "风险"}}',
                     '{{"头实体": "机构","关系类型": "隶属","尾实体": "机构"}}',
                     '{{"头实体": "机构","关系类型": "投资","尾实体": "机构"}}',
                     '{{"头实体": "机构","关系类型": "并购","尾实体": "机构"}}',
                     '{{"头实体": "机构","关系类型": "客户","尾实体": "机构"}}',
                     '{{"头实体": "人物","关系类型": "任职于","尾实体": "机构"}}',
                     '{{"头实体": "人物","关系类型": "投资","尾实体": "机构"}}',
                     '{{"头实体": "机构","关系类型": "生产销售","尾实体": "产品"}}',
                     '{{"头实体": "机构","关系类型": "采购买入","尾实体": "产品"}}',
                     '{{"头实体": "机构","关系类型": "开展","尾实体": "业务"}}',
                     '{{"头实体": "机构","关系类型": "拥有","尾实体": "品牌"}}',
                     '{{"头实体": "产品","关系类型": "属于","尾实体": "品牌"}}',
                     '{{"头实体": "机构","关系类型": "发布","尾实体": "文章"}}']
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''你现在需要完成一个关系抽取任务，定义的关系三元组有'''.join(relation_list) +
             '''抽取的关系请用json的形式展示，其中json的第一个元素为头实体名称，第二个元素为关系类型，第三个元素为尾实体。如果该句子中不含有指定的关系类型，你可以输出: []。除了这个列表以外请不要输出别的多余的话。'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=16)
    chain = LLMChain(
        prompt=final_prompt,
        # 温度调为0，可以保证输出的结果是确定的
        llm=ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1")
        # output_parser=output_parser
    )
    merged_json = {"relation_list": []}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=text, return_only_outputs=True)
                # tmp = chain(
                #     {"input": text}, return_only_outputs=True)['text']
                # print(tmp)
                try:
                    json_object = json.loads(tmp)
                    merged_json = merge_json(merged_json, json_object)
                except Exception as e:
                    continue
            pbar.update(1)
    return merged_json
