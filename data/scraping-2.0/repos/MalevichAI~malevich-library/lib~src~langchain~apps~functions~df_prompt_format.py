import concurrent.futures

import pandas as pd
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage


def df_prompt_format(
    df: pd.DataFrame,
    prompt_template: str,
    chat: BaseChatModel
) -> pd.DataFrame:

    def process_data(data: dict) -> str:
        message = HumanMessage(content=prompt_template.format(**data))
        content = chat([message]).content
        return content

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_data, df.to_dict('records')))

    return pd.DataFrame({"result": results})
