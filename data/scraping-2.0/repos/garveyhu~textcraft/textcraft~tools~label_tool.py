import json
from http import HTTPStatus

import dashscope
from dashscope import Understanding
from langchain.tools import BaseTool

from textcraft.core.config import keys_qwen


class LabelTool(BaseTool):
    name = "标签工具"
    description = "使用Ali的understanding给文本打上标签"
    labels = ""

    def _run(self, text: str, run_manager=None) -> str:
        return self.run_for_label(text)

    async def _arun(
        self,
        text: str,
        run_manager=None,
    ) -> str:
        pass

    def run_for_label(self, text):
        return self.get_label(text, self.labels)

    def get_label(self, text, labels):
        dashscope.api_key = keys_qwen()
        response = Understanding.call(
            model="opennlu-v1",
            sentence=text,
            labels=labels,
            task="classification",
        )

        if response.status_code == HTTPStatus.OK:
            print(json.dumps(response.output, indent=4, ensure_ascii=False))
            return response.output["text"]
        else:
            print(
                "Code: %d, status: %s, message: %s"
                % (response.status_code, response.code, response.message)
            )


if __name__ == "__main__":
    understand = LabelTool()
    sentence = "某股份制银行推出的1年期、2年期、3年期的礼仪存单，利率分别是2.25%、2.85%、3.50%，每个期限的礼仪存单利率只比同期限的大额存单利率低0.05个百分点。"
    labels = "存款，基金，债券，股票，保险"
    label = understand.get_label(sentence, labels)
    print(label)
