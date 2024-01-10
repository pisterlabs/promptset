import yaml
import requests
import os
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import SeleniumURLLoader


from dotenv import load_dotenv

load_dotenv()
CONFIG = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)


class CustomGeminiLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        base_url = os.environ.get(
            'GOOGLE_GEMINI_PROXY', 'https://generativelanguage.googleapis.com')
        url = f"{base_url}/v1beta/models/gemini-pro:generateContent?key={os.environ.get('GOOGLE_API_KEY')}"
        data = {"contents": [
            {"parts": [{"text": prompt}]}
        ]
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers)
        res_data = response.json()
        res = res_data['candidates'][0]['content']['parts'][0]['text']
        return res


def get_title(msg):
    llm = CustomGeminiLLM(model='gemini-pro',)
    prompt = PromptTemplate(
        input_variables=["query"],
        template="给这段文字起一个简短的标题,要少于60个汉字 文字:<<{msg}>>",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain({'msg': msg}).get('text')
    return output


def url_ask_google_genai(msg, url):
    wiki_items = CONFIG.get('docs')
    item_names = ','.join(["["+item.get('name')+"]" for item in wiki_items])

    loader = PlaywrightURLLoader(
        urls=[url], remove_selectors=["header", "footer"])

    data = loader.load()

    url_content = data[0].page_content.replace('{', '').replace('}', '')
    template = """ 你是一位专业的信息整理专家，擅长对信息进行分类，打标签，拟标题。
    下面是你要处理的信息:
    <<{message}   链接的访问:"""+url_content+""">>

    你需要对这条信息进行 1.分类 2.打标签 3.摘要 并且严格按照指定的json格式返回给我
    {format_instructions}
    
    """
    response_schemas = [
        ResponseSchema(
            name="title", description="标题: 给出这段信息的标题,可以是原文的标题,也可以是根据内容拟定的标题,拟定标题的时候可以按照<<xx是xx,主要应用在xx方面'的格式>>"),
        ResponseSchema(
            name="category", description="分类: 判断该条信息属于分类列表 " + item_names + " 中的那一项,返回的json的  value为对应的分类项,注意只返回分类列表中存在的分类"),
        ResponseSchema(
            name="summary", description="摘要: 对这个信息按照要点进行摘要,字数在500个字符内,解释清楚 ‘它是什么’，‘它的原理是什么’，‘它可以应用在那些方面’ 等问题"),
        ResponseSchema(
            name="inspire", description="一个IT或者AI行业从业者可以从这条信息中得到什么启发,有什么好处,获得什么,300个字符"),
        ResponseSchema(
            name="image", description="图片: 链接的访问结果中，最能体现主题的图片的链接"),

    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)
    format_instructions = output_parser.get_format_instructions()
    # print(format_instructions)
    prompt = PromptTemplate(
        input_variables=["message"],
        template=template,
        partial_variables={"format_instructions": format_instructions})

    llm = CustomGeminiLLM(model='gemini-pro')
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain({'message': msg}).get('text')
    res = output_parser.parse(output)
    for i in wiki_items:
        if i['name'] == res['category']:
            res['document_id'] = i['id']
    res['url'] = url
    res['msg'] = msg
    # if len(msg) > 60:
    #     res['title'] = get_title(msg)
    # else:
    #     res['title'] = msg
    return res


def msg_ask_google_genai(msg):
    wiki_items = CONFIG.get('docs')
    item_names = ','.join(["["+item.get('name')+"]" for item in wiki_items])

    template = """ 你是一位专业的信息分类助手，擅长对信息进行分类，打标签，摘要。
    下面是我的信息:
    <<{message}>>

    你需要对这条信息进行 1.分类 2.打标签 3.摘要 并且严格按照指定的json格式返回给我
    {format_instructions}
    
    """
    response_schemas = [
        ResponseSchema(
            name="category", description="分类: 判断该条信息属于分类列表 " + item_names + " 中的那一项,返回的json的  value为对应的分类项,注意只返回分类列表中存在的分类"),
        ResponseSchema(
            name="summary", description="摘要: 对这个信息按照要点进行摘要,字数在200个汉字内,主要解释“是什么”，“为什么”，“怎么做” 等问题"),

    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)
    format_instructions = output_parser.get_format_instructions()
    # print(format_instructions)
    prompt = PromptTemplate(
        input_variables=["message"],
        template=template,
        partial_variables={"format_instructions": format_instructions})

    # print(prompt)
    llm = CustomGeminiLLM(model='gemini-pro')
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain({'message': msg}).get('text')

    res = output_parser.parse(output)
    res['document_id'] = ''
    for i in wiki_items:
        if i['name'] == res['category']:
            res['document_id'] = i['id']
    if not res['document_id']:
        res['document_id'] = wiki_items[-1].get('id')
    res['msg'] = msg
    if len(msg) > 60:
        res['title'] = get_title(msg)
    else:
        res['title'] = msg
    return res


if __name__ == '__main__':
    msg = "调用openai 3.5的需要代理的可以用这个项目直接部署，帮你用node在中间代理了一层，免费安全，亲测可用https://github.com/51fe/openai-proxy"
    # url = "https://mp.weixin.qq.com/s/ILzpMWW8M4ELj66Bfh8P7w"
    # url_ask_google_genai(msg, url)
    res = msg_ask_google_genai(msg)
    print(res)
