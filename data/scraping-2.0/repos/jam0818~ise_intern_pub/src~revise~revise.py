import os
import logging
import json
from datetime import datetime
from openai import OpenAI

logging.basicConfig(filename='reviser.log', level=logging.INFO)


class Reviser:
    """
   revise class
    """

    def __init__(
            self,
            data_path: str,
            save_path: str,
            target_dir: str,
    ) -> None:
        """
        :param data_path: str
        :param save_path: str
        """

        self.data_path = data_path
        self.save_path = save_path
        self.target_dir = target_dir
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
       
        self.logger = logging.getLogger('Reviser')
        self.logger.addHandler(logging.StreamHandler())

    def revise(self, file_name: str) -> dict:
        """
        revise text
        """
        self.logger.info("Revising: " + file_name)

        with open(os.path.join(self.data_path, self.target_dir, file_name), 'r', encoding='UTF-8') as json_file:
            data = json.load(json_file)
            
        # extract integrated data
        fulltext = ''
        for datum in data:
            fulltext += datum.get('text')+'。'

        if fulltext is None:
            raise ValueError('JSONファイルに"test"フィールドが存在しません。')

        # gpt apiに投げるprompt用意
        prompt = '以下の文章の文法をチェックして、見やすくして適宜に改行入れて出力してください。\n' + fulltext + '\n出力:'

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0
        )
            
        revised_text = response.choices[0].message.content

        self.save_revised_text(revised_text)  # save revised text as "revised_integrated.json"
        return revised_text

    def save_revised_text(self, revised_text) -> None:
        """
        save revised integrated text
        """
        self.logger.info("Integrated revise texts in " + self.target_dir)

        if not os.path.exists(os.path.join(self.save_path, self.target_dir)):
            os.mkdir(os.path.join(self.save_path, self.target_dir))

        with open(os.path.join(self.save_path, self.target_dir, "revised_integrated.json"), "w") as f:
            json.dump({"text": revised_text}, f, ensure_ascii=False)

