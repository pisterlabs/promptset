import pandas as pd
import os
import json
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = OpenAI(
    temperature=0,
    model_name="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def extract_contact_name(bankstatements: list[str]) -> list[str]:
    # 先将原始list划分为每50个str一组的小list
    batches = [bankstatements[i : i + 50] for i in range(0, len(bankstatements), 50)]

    result = []
    for batch in batches:
        # 对每个小list调用extract_contact_name_from_llm函数，并将结果添加到结果list中
        result.extend(extract_contact_name_from_llm(batch))
    return result


def extract_contact_name_from_llm(bankstatements: list[str]) -> list[str]:
    print(bankstatements)
    resp = llm(
        """
    帮我从以下银行流水列表中截取出对手方的名称列表，名称全部转化为大写，直接返回JSON的str列表格式即可。
    如果识别不出来，则直接返回空字符串。
    如果银行流水中是No transaction description，则直接返回空字符串。
    银行流水列表如下：
    """
        + str(bankstatements),
    )
    # 解析 JSON 字符串为 Python 对象
    result = json.loads(resp)
    return result


if __name__ == "__main__":
    list = [
        "VAMP SOCIAL LTD. ref: JUNE 22 JUNE 22 ",
        "NISA LOCAL - 10 WOOLWICH NEW ROAD, LONDON  ",
    ]
    extract_contact_name(list)
