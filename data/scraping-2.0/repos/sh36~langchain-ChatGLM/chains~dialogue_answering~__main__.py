import sys
import os
import argparse
import asyncio
from argparse import Namespace

# 将上级目录添加到模块搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

# 导入对话回答相关的模块
from chains.dialogue_answering import *
from langchain.llms import OpenAI
from models.base import (BaseAnswer, AnswerResult)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

# 异步函数，用于处理任务分发
async def dispatch(args: Namespace):
    # 将参数转换为字典形式
    args_dict = vars(args)
    # 设置共享的检查点加载器
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # 加载LLM模型
    llm_model_ins = shared.loaderLLM()
    # 检查对话文件路径是否存在
    if not os.path.isfile(args.dialogue_path):
        raise FileNotFoundError(f'Invalid dialogue file path for demo mode: "{args.dialogue_path}"')
    # 创建OpenAI类实例，温度为0，用于生成回答
    llm = OpenAI(temperature=0)
    # 创建对话实例，包括零-shot react LLM和问答LLM模型
    dialogue_instance = DialogueWithSharedMemoryChains(zero_shot_react_llm=llm, ask_llm=llm_model_ins, params=args_dict)

    # 运行对话代理链，输入问题进行回答
    dialogue_instance.agent_chain.run(input="What did David say before, summarize it")


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser.add_argument('--dialogue-path', default='', type=str, help='dialogue-path')
    parser.add_argument('--embedding-model', default='', type=str, help='embedding-model')
    
    # 解析命令行参数
    args = parser.parse_args(['--dialogue-path', '/home/dmeck/Downloads/log.txt',
                              '--embedding-mode', '/media/checkpoint/text2vec-large-chinese/'])

    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    # 设置事件循环
    asyncio.set_event_loop(loop)
    # 运行任务分发函数
    loop.run_until_complete(dispatch(args))
