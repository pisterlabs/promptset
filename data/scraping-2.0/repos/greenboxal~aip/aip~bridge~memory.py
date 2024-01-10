import requests

from langchain.schema import BaseMemory
from pydantic import BaseModel
from typing import List, Dict, Any
from jsonrpcclient import request, parse, Ok, Error

class BridgeMemory(BaseMemory, BaseModel):
    memory_key: str = "memory"
    endpoint: str = "http://localhost:30100/v1/rpc"

    _current_memory: any = None

    def clear(self):
        pass

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        if self._current_memory is None:
            return {self.memory_key: ""}

        result = self._send_request("memlink.OneShotGetMemory", {
            "memory_id": self._current_memory["metadata"]["id"],
        })

        self._current_memory = result["memory"]

        return {self.memory_key: self._current_memory["data"]["text"]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}

        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]

        page_content = "\n".join(texts)

        request = {
            "new_memory": {
                "text": page_content,
            }
        }

        if self._current_memory is not None:
            request["old_memory"] = self._current_memory

        result = self._send_request("memlink.OneShotPutMemory", request)

        self._current_memory = result["new_memory"]

    def _send_request(self, method, req):
        print(req)
        payload = request(method, params=req)
        print(payload)
        result = requests.post(self.endpoint, json=payload).json()
        print(result)
        parsed = parse(result)
        print(parsed)

        if isinstance(parsed, Error):
            raise Exception(parsed.message)

        return parsed.result
