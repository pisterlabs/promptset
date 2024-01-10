from janome.tokenizer import Tokenizer
from asr_system.repository.client import OutlineClient
from asr_system.repository.client import DispacherClient
from asr_system.repository.file_io import FileIO
from typing import List
from os import getenv
from dotenv import load_dotenv
import openai

load_dotenv()


def split_text_into_chunks(text, max_tokens=1000):
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(text, wakati=True))  # wakati=True returns a list of words
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(''.join(chunk))
    return chunks


def attach_mark(input_list):
    openai.api_key = getenv("OPENAI_API_KEY")
    # text = ''.join(input_list)
    # textlist = []
    gpt_textlist = []
    # textlist = split_text_into_chunks(text)
    for i in range(len(input_list)):
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "条件をもとに、ユーザーからの入力文に対して、**句読点のみ**をつけてください"},
                {"role": "system", "content": "条件1:**文字を補完しない**でください"},
                {"role": "system", "content": "条件2:あなたは**句読点だけ**をつける機械です。**元の文章は変更しない**でください"},
                {"role": "system", "content": "条件3:**文字を補完しない**でください"},
                {"role": "system", "content": "条件4: 私はえをが好きだ → 私は、えを、が好きだ のように文字を補完せず出力してください"},
                {"role": "system", "content": "条件5:句読点をつける場合、「、」か「。」のみを使用してください"},
                {"role": "system", "content": "条件6:音声入力によって得られたテキストなので不自然かもしれませんが、自然に句読点をつけてください"},
                {"role": "system", "content": "条件7:**口語も存在するので**、それを考慮しながら、自然に句読点をつけてください"},
                {"role": "system", "content": "条件8:単語や文の区切りを意識してください"},
                {"role": "system", "content": "条件9:*不自然に文が終わる場合、「。」をつけてください*"},
                {"role": "system", "content": "条件10:句読点をつける必要がない場合、入力されたままの文章を返してください"},
                {"role": "system", "content": "条件11:**入力がないときは、句読点をつけず、改行だけをしてください**"},
                {"role": "user", "content": input_list[i]},
            ],
        )
        gpt_textlist.append(response.choices[0]["message"]["content"].strip())
    return gpt_textlist
    # gpt_text = ''.join(gpt_textlist)
    # gpt_text_eracemark = re.sub(r'[、。]','',gpt_text)
    # print("補完率",+ cer(gpt_text_eracemark.replace('\n',''),text.replace('\n','')))
    # return gpt_text


class TextHandler:
    def __init__(self) -> None:

        self.outline_client = OutlineClient()
        self.dispatcher_client = DispacherClient()
        self.local_output_path = getenv("TEXT_OUTPUT")

    def initialize_document(self, title: str):
        self.document_id = self.outline_client.create_document(title)

    def write_text(self, text_list: List, file_name: str):
        file_path = f"{self.local_output_path}/{file_name}"
        FileIO.output_text_file(text_list, file_path)

    # notify to dispatcher api
    def notify_to_dispatcher(self, file_name):
        file_path = f"{self.local_output_path}/{file_name}"
        self.dispatcher_client.nofity_finish_send_text(file_path)

    def send_text_outline(self, text: str):
        self.outline_client.update_document(text, self.document_id)

    def send_text_list_to_outline(self, text_list: List[str]):
        self.outline_client.overwrite_document(text_list, self.document_id)

    def final_send_text_outline(self, text_list: List[str]):
        self.outline_client.final_update(text_list, self.document_id)
