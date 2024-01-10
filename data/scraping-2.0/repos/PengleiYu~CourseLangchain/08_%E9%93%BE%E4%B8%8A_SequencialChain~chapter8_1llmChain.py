from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI

llm = OpenAI(temperature=0)
prompt_template = PromptTemplate.from_template('{flower}在{season}的花语是?')
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 直接执行会返回原始字典
# result = llm_chain(inputs={'flower': "玫瑰", 'season': "夏季", })

# run和predict只返回响应文本
# result = llm_chain.run({'flower': "玫瑰", 'season': "夏季", })
# result = llm_chain.run(flower='玫瑰', season='夏季')
# result = llm_chain.predict(flower='玫瑰', season='夏季')


input_list = [{"flower": "玫瑰", 'season': "夏季"},
              {"flower": "百合", 'season': "春季"},
              {"flower": "郁金香", 'season': "秋季"}]
# 应用到多个输入上
# result = llm_chain.apply(input_list=input_list)
# 同上，但返回result对象包含更多信息
result = llm_chain.generate(input_list=input_list)
print(result)
