from langchain.document_loaders import UnstructuredURLLoader # url 加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter # 文本分割器
from langchain.chains.summarize import load_summarize_chain # 总结链
from langchain.llms import OpenAI
from dotenv import load_dotenv # 环境变量加载器
from langchain.prompts import PromptTemplate # 提示词模板
from langchain.chat_models import ChatOpenAI # 对话模型
from langchain.schema import HumanMessage # 人类信息（⚠️使用聊天模型时候需要引入！）
from langchain.output_parsers import PydanticOutputParser # 输出解析器
from pydantic import BaseModel, Field # 🌟从输出解析器中引入 BaseModel 和 Field 类
from typing import Union


load_dotenv() # 加载环境变量



class TalkShow_line(BaseModel):
    character: str = Field(description="说这句台词的角色名称")
    content: str = Field(description="台词的具体内容, 其中不再包含角色名字")
    
class TalkShow(BaseModel):
    script: list[TalkShow_line] = Field(description="脱口秀台词的剧本")
    
    

    

# 🌟 【一】提取新闻内容 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def urlToNews(URL) -> str:
	text_splitter = RecursiveCharacterTextSplitter(  # 文本分割器
		separators = ["正文", "撰稿"], # 分割头尾
		chunk_size = 1000, # 每个分割块的大小
		chunk_overlap = 20, # 重叠部份
		length_function = len # 分割长度(len 为默认)
	)

	urlLoader = UnstructuredURLLoader([URL]) # url 加载器
	# data = urlLoader.load() # 普通加载
	data = urlLoader.load_and_split(text_splitter=text_splitter) # 使用文本分割器加载数据 (返回新闻列表数据, 包含了新闻主体)
	# print(data)
	# print(data[1:2])
	return data[1:2] # 表示返回的范围是从第 1 个到第 2 个(不包含), data[1:2] 是一个列表切片操作。这个操作会从列表 data 中选取索引范围为 1 到 2（不包括 2）的元素




# 🌟 【二】进行总结 => 利用 langchain 的总结链 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	# stuff 的总结链模式（把所有文档合并起来作为上下文, 然后结合提示词发给 llm） ❌不推荐
	# map reduce 模式 (把每个文档分别都发给 llm 然后分别得到总结, 最后合并总结成为总结的上下文再结合提示词发给 llm) ✅推荐
def content_summary(llm, URL) -> str:
	# 👇根据源码改写的总结新闻的 prompt
    prompt_template = """总结这段新闻的内容:
		"{text}"
	新闻总结:"""
 
    chn_prompt = PromptTemplate(template=prompt_template, input_variables=["text"]) # 用自定义的 prompt 模板来进行总结
 
	# refine 模式 (不停的随机拿个文档发给 llm, 不停的比较每个文档所产生的答案, 最终得到一个最好的答案) ✅推荐
    summary_chain = load_summarize_chain(llm, prompt=chn_prompt) # 总结链, 传入 llm 和 prompt
    doc_content = urlToNews(URL) # 拿到的网页内容
    summary = summary_chain.run(doc_content)# 把拿到的内容喂给总结链
    # print(summary)
    return summary
    
    


# 🌟 【三】把拿到的 summary 转为脱口秀 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def tranTo_talkshow(summary) -> TalkShow | str: # -> Union[TalkShow, str] 联合类型, 表明可能返回 TalkShow 对象或一个字符串
    openAI_chat = ChatOpenAI(model_name="gpt-3.5-turbo") # 选择 AI 的对话模型
    
    # 👇提示词模板, {要求} 为脱口秀的风格, {output_instructions} 为告诉大模型要以什么样的结果进行输出(返回序列化的文本), 以满足下方【文本解析器】的要求(下方的解析器再把文本进行序列化)
    template = """\
		我将给你一段新闻的概括, 请按照要求把这段新闻改成李诞和郭德纲脱口秀剧本。
  
		新闻: "{新闻}"
		要求: "{要求}"
		{output_instructions}
    """
    parser = PydanticOutputParser(pydantic_object=TalkShow)    
     
    # 这个方法的目的是根据提供的示例来创建一个新的 PromptTemplate 实例, 用来定义一个具体的情境或格式，然后该模板可以用来生成特定风格或格式的文本提示
    # prompt_talkShow = PromptTemplate.from_template(template=template) # ⚠️不带【部分参数 output_instructions】 以及 parser 解析器的写法
    prompt_talkShow = PromptTemplate(
		template=template,
		input_variables=["新闻", "要求"], # 🌟 告诉 llm 说【新闻】跟【要求】不是部分参数
		partial_variables={"output_instructions": parser.get_format_instructions()} # 🌟【部分参数】, 值直接从 parser 解析器中拿到
	)
    
    # 人类的信息输入
    human_msg = [HumanMessage(
        content=prompt_talkShow.format( # 传入 prompt_talkShow, 并进行格式化以及传参
            新闻=summary, 
            要求="风趣幽默, 带有社会讽刺意味, 剧本对话角色分别为李诞和郭德纲, 以他们的自我介绍为开头"
		))
	]
    
    
    # AI 输出的结果
    content_script = openAI_chat(human_msg)
    # print(content_script.content)
    
    # 调用文本解析器, 把 AI 输出的结果进行序列化
    talkShow_content = parser.parse(content_script.content) # 把 AI 输出的结果进行序列化
    return talkShow_content # 最终返回 script=[TalkShow_line(character='李诞', content='大家好，我是李诞！'), TalkShow_line(character='郭德纲', content='大家好，我是郭德纲！'), ...] 的序列化结构


# 🌟入口函数 (供外部调用)
def convertToTalkshow(URL) -> str:
    llm = OpenAI(max_tokens=500) # 🌟用大语言来进行总结, 默认的 token 为 256, 可以扩充更多一些
    summary = content_summary(llm, URL)
    res = tranTo_talkshow(summary)
    # print(res)
    return res # 最终返回 script=[TalkShow_line(character='李诞', content='大家好，我是李诞！'), TalkShow_line(character='郭德纲', content='大家好，我是郭德纲！'), ...] 的序列化结构
    
    


# # 🌟 主函数
# if __name__ == '__main__':
#     URL = "https://news.sina.com.cn/c/2023-08-02/doc-imzetmzi8136053.shtml"
#     res = convertToTalkshow(URL)
#     print(res)

    
	














