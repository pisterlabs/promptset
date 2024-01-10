from openai import OpenAI
from uuid_extensions import uuid7str
import dotenv
import os

# APIキーの設定
dotenv.load_dotenv()
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


class OpenAIAdapter:
    def __init__(self, file_root) -> None:
        self.file_root = file_root
        self.output_directory_name = os.environ.get(
            "OUTPUT_DIRECTORY_NAME", "output")
        # ディレクトリがなければ作る
        if not os.path.exists(self.output_directory_name):
            os.mkdir(self.output_directory_name)

        # system_promptはsystem_prompt.txtから読み込む
        with open("system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        pass

    def _create_message(self, role, message):
        return {
            "role": role,
            "content": message
        }

    def create_chat(self, question):
        system_message = self._create_message("system", self.system_prompt)
        user_message = self._create_message("user", question)
        messages = [system_message, user_message]
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages)
        # 返り値のテキストを出力する
        content = res.choices[0].message.content

        # ファイルに書き出す
        text_file_path = f"{self.output_directory_name}/{self.file_root}-poll.txt"
        with open(text_file_path, mode="w", encoding="utf-8") as f0:
            f0.write(content)

        return content


if __name__ == "__main__":
    uuid = uuid7str()
    adapter = OpenAIAdapter(uuid)
    response_text = adapter.create_chat("こんにちは")
    print(response_text)
