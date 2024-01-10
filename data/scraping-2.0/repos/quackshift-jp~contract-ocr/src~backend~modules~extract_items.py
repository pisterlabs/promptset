import json

from openai import OpenAI

from backend.models.item_extractor import ItemExtractor


class OpenaiItemExtractor(ItemExtractor):
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def extract_items(self, text) -> dict[str, dict[str, any]]:
        system_prompt = """
        あなたは契約書から項目を読み取るアシスタントです。
        与えられた文字列に対して、物件名と住所をJSON形式でパースしてください。
        JSONのキーはname, locationとしてください。

        nameは物件名で、文字列オブジェクトです。
        locationは住所で、文字列オブジェクトです。

        抽出できなかった項目に関しては、空のバリューを返してください。
        """

        prompt = f"""
        次の入力を、所定のJSONフォーマットで出力してください。
        - [入力]
        {text}

        - [出力JSONフォーマット]
        {{
            "content":{{
                "物件名": *(str型)
                "住所": *(str型)
            }},
        }}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
