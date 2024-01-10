import os

from openai import OpenAI

from schema.data import SelectData
from utils.logging import logger


def get_llm_response(model: str, content: str) -> str:
    client = OpenAI(base_url=os.getenv('LLM_URL'), api_key=os.getenv('API_KEY'))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
        max_tokens=512,
    )
    logger.debug(completion.choices[0].message.content)
    return completion.choices[0].message.content
    
    
def get_llm_select_result(model: str, select_data: SelectData) -> bool:
    content = f"请你解决下面的选择题：\n{select_data.question}\nA.{select_data.A}\nB.{select_data.B}\nC.{select_data.C}\nD.{select_data.D}\n请你只输出代表答案序号的一个大写英文字母（A、B、C或D），禁止输出选项内容和判断理由。"
    logger.debug(content)
    llm_answer: str = get_llm_response(model, content)
    option_index = [llm_answer.rfind(x) for x in list("ABCD")]
    return option_index.index(max(option_index)) == select_data.answer
