from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from .parsingTool import letter_parser, TextParsing


def generate_letter(person_info: str) -> TextParsing:
    letter_template = """
    下面是这个人的微博信息
    {information}
    请你帮我: 
    1. 写一个简单的总结 
    2. 挑两件有趣的事情说一说 
    3. 找一些他比较感兴趣的事情 
    4. 写一篇热情洋溢的介绍信
    {format_instructions}
    """
    prompt_template = PromptTemplate.from_template(
        letter_template,
        partial_variables={
            "format_instructions": letter_parser.get_format_instructions(),
        })
    llm = ChatOpenAI(model_name='gpt-4-1106-preview')
    llm_chain = LLMChain(prompt=prompt_template, llm=llm, verbose=True)
    response = llm_chain.run(information=person_info)
    print('llm response=', response)
    parse_result = letter_parser.parse(response)
    print('parse_result = ', parse_result)
    return parse_result
