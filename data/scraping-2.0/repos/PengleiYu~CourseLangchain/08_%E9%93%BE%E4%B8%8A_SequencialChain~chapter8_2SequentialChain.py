import json

from langchain.llms.openai import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sequential import SequentialChain

llm = OpenAI(temperature=0.7)
introduction_chain = LLMChain.from_string(llm=llm,
                                          template="""
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。

花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:""")
introduction_chain.output_key = 'introduction'

review_chain = LLMChain.from_string(llm=llm, template="""
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。

鲜花介绍:
{introduction}
花评人对上述花的评论:""")
review_chain.output_key = 'review'

social_post_chain = LLMChain.from_string(llm=llm, template="""
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。

鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:""")
social_post_chain.output_key = 'social_post_text'

overall_chain = SequentialChain(chains=[introduction_chain, review_chain, social_post_chain],
                                input_variables=["name", "color"],
                                # output_variables=["introduction", "review", "social_post_text"],
                                return_all=True,
                                verbose=True,
                                )
result = overall_chain(inputs={"name": "玫瑰", "color": "黑色"})
print(result)
