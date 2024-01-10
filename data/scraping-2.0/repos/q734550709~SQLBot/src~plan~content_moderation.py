import openai
from src.get_completion_from_messages import get_completion_from_messages

# 创建审查输入内容的函数，0为正常，>0为异常
def content_moderation (eval_message):
    response = openai.Moderation.create(
    input=eval_message
    )
    moderation_output = sum(1 for value in response["results"][0]["categories"].values() if value)
    return moderation_output

#判断是否查询表信息
def is_query_question(user_message):
    delimiter = "####"
    # 遍历关键词列表，检查文本中是否存在这些关键词
    system_message = f"""
    你是任务是确定用户的输入是否属于查询表信息的文本。
    输入的文本可能包含下面的文字：

    查找，找出，从……查询，取一下，取出，等等类似表述

    或者，英文的表述有：which; find; how many; what is; what are

    用户输入的内容会使用{delimiter}作为分割符,
    使用 Y 或者 N 回复:
    Y - 如果用户的输入属于查询表信息的文本
    N - 其他情况

    只需输出一个字符
    """

    user_message_for_model = f"""
        {delimiter}{user_message}{delimiter}
        """
    messages =  [
    {'role':'system', 'content': system_message},
    {'role':'user', 'content': user_message_for_model},
    ]

    is_query_question = get_completion_from_messages(messages, max_tokens=1)

    return is_query_question
