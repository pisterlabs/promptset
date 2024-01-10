"""
本文件展示了如何使用多模板链中的核心类RouterChain
"""

flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}"""
flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}"""
prompt_infos = [{"key": "flower_care",
                 "description": "适合回答关于鲜花护理的问题",
                 "template": flower_care_template,
                 },
                {"key": "flower_decoration",
                 "description": "适合回答关于鲜花装饰的问题",
                 "template": flower_deco_template,
                 }, ]

from langchain.llms.openai import OpenAI

llm = OpenAI()

from langchain.chains.llm import LLMChain

from langchain.prompts.prompt import PromptTemplate

"""
1,目的链列表
"""
destination_chains = {
    info['key']: LLMChain(prompt=PromptTemplate.from_template(info['template']),
                          llm=llm,
                          verbose=True, )
    for info in prompt_infos
}

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

"""
2,路由链
"""
destinations = [f'{info["key"]}:{info["description"]}' for info in prompt_infos]
# 这个字符串必须format一次，不能直接用于模板。因为其中input外面有两层大括号，format一次后才会去掉一层大括号
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations="\n".join(destinations))
router_prompt = PromptTemplate.from_template(
    template=router_template,
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain(llm_chain=LLMChain(llm=llm, prompt=router_prompt, verbose=True))

from langchain.chains import ConversationChain

"""
3,默认链
"""
# LLMChain需要模板，ConversationChain内部已经有模板了
default_chain = ConversationChain(llm=llm,
                                  output_key='text',  # 默认链必须指定输出的key
                                  verbose=True)

from langchain.chains.router import MultiPromptChain

"""
4,最终的多模板链
"""

from utils.callback_handler import MyBaseCallbackHandler

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True,
                         # callbacks=[MyBaseCallbackHandler()],
                         )
# 注意：callbacks传入run函数和传入构造函数是不同的。run方法中会作用于所有遇到的chain，而构造方法中仅作用于该chain
print(chain.run("如何为玫瑰浇水？", callbacks=[MyBaseCallbackHandler()]))
print(chain.run("如何为婚礼场地装饰花朵？"))
# print(chain.run("如何考入哈佛大学？"))
