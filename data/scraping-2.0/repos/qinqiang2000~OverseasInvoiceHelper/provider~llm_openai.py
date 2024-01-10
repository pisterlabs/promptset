import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()


class LLMOpenAI:
    def __init__(self, model):
        self.model = model

    def generate_text(self, text, sys_prompt, reqid):
        print("使用模型API：", self.model, reqid)
        try:
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}
                ]
            )
        except Exception as e:
            print(f"调用openai出错：{e}")
            return json.dumps({"error": "fail: 调用大模型接口出错"})

        print("total tokens:", response.usage.total_tokens)
        print("system_fingerprint:", response.system_fingerprint)
        print(response.choices[0].message.content)

        return response.choices[0].message.content