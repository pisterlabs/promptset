from typing import Optional, Type
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import requests
import json

url = 'https://api.lawapi-prototype-test-elaws.e-gov.go.jp/api/2'
case_search_url = "https://www.courts.go.jp/app/hanrei_jp/list1?filter[text1]="


# トピック一覧を取得するツール
class ListLawTool(BaseTool): # BaseToolクラスのサブクラスとして、クラスを自作
    name = "ListLawTool"
    description = """
    Get Infomation of Laws by keywords from user.
    If you infer the keywords to search via chatting with the user, use this functionality.
    For example, if the user is searching the law related to 'parking', you should use this tool to search
    laws related to parking.
    """
    
    # エンドポイントにGETリクエストを送信
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        params = {'keyword': query, 'limit': 2}
        response = requests.get(url + "/keyword", params=params)
        items = response.json().get("items")

        if items is None:
            return []

        res = []

        for item in items:
            reduced_item = {}
            reduced_item["law_id"] = item["law_info"]["law_id"]
            reduced_item["law_num"] = item["law_info"]["law_num"]
            reduced_item["sentence"] = item.get("sentence")
            reduced_item["title"] = item.get("revision_info")["law_title"]
            #reduced_item["case_link"] = f'{case_search_url}{item.get("revision_info")["law_title"]}'
            res.append(reduced_item)

        return res
    
    # 非同期実行の関数も必須
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("ListTopicTool does not support async")
