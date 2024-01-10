# 地区抽取
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


async def paper_read(abstract, prompts):
    model = "Qwen-14B-Chat-Int4"
    res = {}
    for prompt in tqdm(prompts, desc="paper_read", total=len(prompts)):
        examples = prompt["example"]
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
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt["system"]),
                few_shot_prompt,
                ("human", "{input}")
            ]
        )
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
        tmp = await chain.arun(input=abstract, return_only_outputs=True)
        res[prompt["mission"]] = tmp
    return res
