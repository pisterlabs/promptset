import polib
import argparse
import deepl
from openai import OpenAI
import os


class DeepL:
    def __init__(self) -> None:
        self.translator = deepl.Translator(os.environ.get("API_KEY"))
    def translation(self,msg,target_lang='zh',glossary=None):
        result = self.translator.translate_text(msg,target_lang=target_lang,split_sentences='nonewlines',glossary=glossary)
        return result.text

class ChatGPT:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.environ.get("API_KEY"))
    def translation(self,msg):
        message = [
            {
                "role": "system",
                "content": f"你是一个翻译引擎，精通从英文到中文的翻译，并且拥有计算机相关领域知识，知道asciidoc格式的规范。\n 请将下面的文本翻译为中文，不需要解释，请直接返回译文。",
            },
            {
                "role": "user",
                "content": msg
            }
        ]
        chat_completion = self.client.chat.completions.create(
            messages=message,
            temperature=0.15,
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机器翻译")
    parser.add_argument('--input', default=None,
                        dest='input', type=str, help='输入文件')
    parser.add_argument('--output', default=None,
                        dest='output', type=str, help='输出文件')
    parser.add_argument('--type', default='deepl',
                        dest='type', type=str, help='翻译类型')
    args = parser.parse_args()
    po = polib.pofile(args.input)
    tr = DeepL() if args.type == 'deepl' else ChatGPT()
    tr = ChatGPT()
    for i in po.untranslated_entries():
        if 'delimited block . 4' not in i.comment:
            i.msgstr = tr.translation(i.msgid)
            i.fuzzy = True
    po.save(args.output)
