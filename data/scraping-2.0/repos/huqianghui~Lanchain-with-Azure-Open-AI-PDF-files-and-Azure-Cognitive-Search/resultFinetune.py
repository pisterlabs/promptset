#Note: The openai-python library support for Azure OpenAI is in preview.
import json
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from openAIRoundRobin import get_openaiByRoundRobinMode


systemTemplate='''You are an accountant, 
and you need to extract the numerical value 
and unit or percentage of the indicators from the provided information based on the indicator type, 
excluding all other content.
'''

few_shot_human1 = HumanMessagePromptTemplate.from_template('''
                                                           指标：结算面积 \n
                                                           指标类型：累计值  \n
                                                           内容：{{  "结算面积": "3,401.4,  
                                                                   "单位": "万平方米"}}''')
few_shot_assistant1=AIMessagePromptTemplate.from_template('''
                                                           3,401.4（万平方米）''')

few_shot_human2 = HumanMessagePromptTemplate.from_template('''
                                                        指标：客户续租率 \n
                                                        指标类型：累计值\n
                                                        内容：{{ "客户续租率": "60%"}}   
                                                        ''')
few_shot_assistant2=AIMessagePromptTemplate.from_template('''60%''')

few_shot_human3 = HumanMessagePromptTemplate.from_template('''
                                                        指标：可租赁建筑面积:物流仓储服务:高标库 \n
                                                        指标类型：期末值 \n
                                                        内容：{{ "物流仓储服务": 
                                                                    {{"高标库": {{
                                                                    "可租赁建筑面积": "846万平方米",
                                                                    "稳定期出租率": "90%"}},
                                                                "冷链园区": 
                                                                    {{"可租赁建筑面积": "118万平方米",
                                                                     "稳定期使用率": "75%"}},
                                                                     "经营收入": 
                                                                            {{ "总收入": "35.6亿元",
                                                                              "高标库营业收入": "21.6亿元",
                                                                              "冷链营业收入": "14.0亿元"}}\n  }}\n}} 
                                                        ''')
few_shot_assistant3=AIMessagePromptTemplate.from_template('''846(万平方米)''')

system_message_prompt = SystemMessagePromptTemplate.from_template(systemTemplate)

user_message_prompt = HumanMessagePromptTemplate.from_template(''' 指标：{index} \n
                                                           指标类型：{index_type} \n
                                                           内容：{content}''')

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    few_shot_human1,few_shot_assistant1,
    few_shot_human2,few_shot_assistant2, 
    few_shot_human3,few_shot_assistant3,
    user_message_prompt])

# 通过llm和few-shot prompt创建一个chain
chain = LLMChain(llm=get_openaiByRoundRobinMode(), prompt=chat_prompt,verbose=True)

def get_finetune_result_by_gpt(index, index_type, content):
    try:
        result = chain.run(index=index, index_type=index_type, content=content)
        print(result)
        return result
    except Exception as e:
        print(content)
        print(e)
        return content
        # # Get all properties from the dictionary
        # try:
        #     jsonResult=json.loads(content)
        #     properitesLen = len(jsonResult.items())
        #     if(properitesLen == 1):
        #         for key, value in jsonResult.items():
        #              return value
        #     if(properitesLen == 2 and "单位" in jsonResult.keys()):
        #         for key, value in jsonResult.items():
        #              if("单位" != key):
        #                 return value + "("+ jsonResult["单位"] + ")"
        #     return content
        # except Exception as e:
        #     print(e)
        #    return content
        
def get_finetune_result(index, index_type, content):
    # Get all properties from the dictionary
    try:
        jsonResult=json.loads(content)
        properitesLen = len(jsonResult.items())
        if(properitesLen == 1):
            for key, value in jsonResult.items():
                    return value
        if(properitesLen == 2 and "单位" in jsonResult.keys()):
            for key, value in jsonResult.items():
                    if("单位" != key):
                        return value + "("+ jsonResult["单位"] + ")"
        return get_finetune_result_by_gpt(index, index_type, content)
    except Exception as e:
        print(e)
        return content

        