from langchain.prompts import PromptTemplate
import openai

template = """
你是业务咨询顾问。
你给一个销售{product}的电商公司，起一个好的名字？
"""

# from_template内部也调用了构造函数
prompt = PromptTemplate.from_template(template)
print(prompt.format(product='水果'))

prompt = PromptTemplate(input_variables=["product", "market"],
                        template="你是业务咨询顾问。对于一个面向{market}市场的，专注于销售{product}的公司，你会推荐哪个名字？")
print(prompt.format(product='鲜花', market='高端'))

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                    {"role": "user", "content": "Who won the world series in 2020?"},
                                                    {"role": "assistant",
                                                     "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                                    {"role": "user", "content": "Where was it played?"}])
print(completion)
