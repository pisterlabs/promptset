import requests
from typing import Optional, List, Dict, Mapping, Any

import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache

# 启动llm的缓存
langchain.llm_cache = InMemoryCache()


class ChatGLM(LLM):

    base_url: str = None
    api_key: str = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _construct_query(self, prompt: str) -> Dict:
        """构造请求体
        """
        query = {"human_input": prompt}
        return query

    def load_config(self, config: Mapping[str, Any]) -> None:
        """加载配置
        """
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]

    @classmethod
    def _post(cls, url: str, key, query: Dict) -> Any:
        """POST请求
        """
        _headers = {"Content_Type": "application/json",
                    "Authorization": f"Bearer {key}"}
        with requests.session() as sess:
            resp = sess.post(url, json=query, headers=_headers, timeout=60)
        return resp

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """_call
        """
        # construct query
        query = self._construct_query(prompt=prompt)

        # post
        res = self._post(url=f"{self.base_url}/v1/chat/completions", key=self.api_key, query=query)

        if res.status_code == 200:
            res_json = res.json()
            predictions = res_json["response"]
            return predictions
        else:
            return "请求模型"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {"base_url": self.base_url, "api_key": self.api_key}
        return _param_dict

