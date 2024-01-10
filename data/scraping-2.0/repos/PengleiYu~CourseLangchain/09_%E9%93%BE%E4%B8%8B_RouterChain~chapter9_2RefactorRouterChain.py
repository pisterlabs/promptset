"""
使用MultiPromptChain.from_prompts来重构路由链的使用
"""
from langchain.chains.router.multi_prompt import MultiPromptChain
from langchain.llms.openai import OpenAI

flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}"""
flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}"""
prompt_infos = [
    {
        'name': 'flower_care',
        'description': '适合回答关于鲜花护理的问题',
        'prompt_template': flower_care_template,
    },
    {
        'name': 'flower_decoration',
        'description': '适合回答关于鲜花装饰的问题',
        'prompt_template': flower_care_template,
    },
]

llm = OpenAI()
chain = MultiPromptChain.from_prompts(llm=llm, prompt_infos=prompt_infos, verbose=True)
print(chain.run("如何为玫瑰浇水？"))
# print(chain.run("如何为婚礼场地装饰花朵？"))
# print(chain.run("如何考入哈佛大学？"))
