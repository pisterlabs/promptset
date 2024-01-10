from typing import Optional, Type
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import requests
import json
from pydantic import Field
import uuid

url = 'https://api.lawapi-prototype-test-elaws.e-gov.go.jp/api/2'


# トピック一覧を取得するツール
class DetailLawTool(BaseTool): # BaseToolクラスのサブクラスとして、クラスを自作
    name = "DetailLawTool"
    description = """
    Get Detail of Law and writes it to text file.
    detail sholud be 
    """
    
    # エンドポイントにGETリクエストを送信
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params = {'law_id': query}
        response = requests.get(url + "/lawdata", params=params)
        law_full_text = response.json().get("law_full_text")
        law_full_text_str = json.dumps(law_full_text)
        uid = uuid.uuid1()
        uid = "tmp"

        # Append the input to the text file
        path = f"{uid}.txt"
        with open(path, 'w') as f:
            f.write(law_full_text_str)

        return path
    
    # 非同期実行の関数も必須
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("ListTopicTool does not support async")
