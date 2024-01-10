import asyncio
import json
import os
from typing import List

from langchain.chains import LLMChain

from server.utils import get_ChatOpenAI, get_prompt_template
from langchain.prompts import Prompt, ChatPromptTemplate
from server.chat.utils import History
from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, logger)


def get_chain(model_name: str = LLM_MODEL,
              prompt_name: str = "question_clustering",
              temperature: float = 0.7,
              ) -> LLMChain:
    model = get_ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        stream=False,
    )
    # （history）
    # 拿到prompt模版
    prompt_template = get_prompt_template(prompt_name)
    # 把prompt模版转换成ChatMessagePromptTemplate（langchain里的对象）
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    # 1.把history里的所有内容模版转换成ChatMessagePromptTemplate，加入消息列表
    # 2.把这次的模板input_msg加入消息列表
    # 3.把消息列表(1和2)一块转换成 ChatPromptTemplate（一个包含所有消息的字符串）
    # （ChatPromptTemplate 是 langchain 库中的一个类，用于生成聊天模板。它接受一个消息列表作为输入，返回一个包含所有消息的字符串。）
    chat_prompt = ChatPromptTemplate.from_messages([input_msg])
    # 创建 chain
    chain = LLMChain(prompt=chat_prompt, llm=model)
    return chain


def cluster_questions_with_chatgpt(questions):
    # 初始化 ChatGPT
    chain = get_chain()

    # 存储最终的分类结果
    categorized_questions = {}
    reference_categories = []

    # 分批处理问题
    for batch in chunk_questions(questions):
        response_dir = None

        # 构建 prompt
        # task = asyncio.create_task(
        #     chain.acall({"questions": "\n".join(batch), "Reference_Categories": "\n".join(reference_categories)}))

        # 使用 ChatGPT 进行处理
        try:
            response = asyncio.run(
                chain.acall({"questions": "\n".join(batch), "Reference_Categories": "\n".join(reference_categories)}))
        except Exception as e:
            logger.error(f"task failed with error: {e}")
            return
        else:
            # Continue with the rest of the code that depends on task1's result
            if response is None or response == {}:
                logger.error(f"chatgpt response is empty")
                return
            else:
                if not is_valid_json_format(response['text']):
                    logger.error(f"chatgpt response is not in valid format")
                    return
                response_text = response['text']
                # Load the JSON string into a Python dictionary
                response_dir = json.loads(response_text)

            # 解析并存储回答
            categorized_questions = update_categorized_questions(response_dir, categorized_questions)

            # 更新参考类别列表
            reference_categories = list(categorized_questions.keys())

    return categorized_questions


def chunk_questions(questions, chunk_size=25):
    # 将问题分批，每批数量不超过 ChatGPT 的限制
    for i in range(0, len(questions), chunk_size):
        yield questions[i:i + chunk_size]


def is_valid_json_format(s):
    try:
        # 尝试将字符串解析为JSON
        data = json.loads(s)

        # 检查外层结构是否为字典
        if not isinstance(data, dict):
            return False

        # 遍历外层字典
        for key, value in data.items():
            # 检查外层的键是否为字符串
            if not isinstance(key, str):
                return False

            # 检查值是否为字典
            if not isinstance(value, dict):
                return False

            # 遍历内层字典
            for inner_key, inner_value in value.items():
                # 检查内层的键是否可以解析为整数
                try:
                    int(inner_key)
                except ValueError:
                    return False

                # 检查内层的值是否为字符串
                if not isinstance(inner_value, str):
                    return False

        # 如果所有检查都通过，则格式正确
        return True
    except json.JSONDecodeError:
        # 如果无法解析为JSON，则格式不正确
        return False


def update_categorized_questions(response_dir, categorized_questions):
    # 解析 ChatGPT 的回答并更新分类结果
    categories = parse_response(response_dir)
    for category, questions in categories.items():
        if category in categorized_questions:
            categorized_questions[category].extend(questions)
        else:
            categorized_questions[category] = questions
    return categorized_questions


def parse_response(response_dir):
    # 解析 ChatGPT 的 JSON 回答并提取分类结果
    # 这部分需要根据 ChatGPT 返回的具体格式进行调整
    parsed_response = {}
    # 示例代码，根据实际返回结果调整
    for category, questions in response_dir.items():
        parsed_response[category] = list(questions.values())
    return parsed_response


def get_question_list(selected_kb):
    # Construct the file path
    file_path = "../../knowledge_base/" + selected_kb + "/qa.txt"
    file_path = os.path.join('knowledge_base', selected_kb, 'qa.txt')
    with open(file_path, 'r') as file:
        # Reading all lines and excluding the ones that are just asterisks
        lines = [line.strip() for line in file if line.strip() != '***']
    return lines


def get_stat_result(knowledge_base_name: str = "", ):
    question_list = get_question_list(knowledge_base_name)
    clustered_questions = cluster_questions_with_chatgpt(question_list)
    print(clustered_questions)
    return clustered_questions

# get_stat_result("test")
