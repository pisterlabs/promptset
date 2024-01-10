# coding=utf-8
"""
1 入力された日本語をdeeplで英語に翻訳
2 1をgpt3.5-turboにprompt付きで投入
3 2の出力をdeeplで日本語に翻訳
4 3の出力を表示
"""
import pathlib
import openai
from utils import get_config
from utils.deepl_wrapper import DeepL
from utils.openai_wrapper import ChatGPT
base_path = pathlib.Path.cwd().parent.parent
config_file = base_path / 'config' / 'config.yaml'

config = get_config.run(config_file)
openai.api_key = config.openai_api_key
deepl_api_key = config.deepl_api_key


prompt_template = """
# introduction
- You are a helpful assistant that provides advice on making a resume more attractive.
- The user is trying to create a resume, and your goal is to make the user's resume attractive.
- Please output the best results based on the following constrains

# Constrains
- Translate the following work experience into a more attractive and engaging description.
- Make the subject of your answer me.
- If you cannot provide the best information, let me know.

Human: {input}
Assistant:
"""


def main():
    # 1 入力された日本語をdeeplで英語に翻訳
    input_text = """
    私は株式会社Aにおいて、webサイトの行動履歴を用いて、どのような行動をしているユーザーがアクションをする割合を算出し、
    予測したアクション確率によって支援の内容を変えるという施策を実施してきました。
    また、アイテムの推薦システムのアルゴリズムの開発を行い、
    recboleやbert4recを用いて推薦するアルゴリズムの検証・システムの構築・効果検証を実践してきました。
    """
    print(input_text)

    deepl = DeepL(config.deepl_api_key, config.deepl_url)
    translate_result = deepl.run_translate(
        text=input_text,
        from_lang='JA',
        to_lang='EN'
    )

    print(translate_result)

    # 2 gpt3.5-turboにprompt付きで投入
    chatgpt = ChatGPT(config.openai_api_key, config.openai_model)
    prompt = prompt_template.format(**{'input':translate_result})
    gpt_response = chatgpt.run_chat(prompt)

    print(gpt_response)

    # 3 deeplで日本語に翻訳
    translate_result = deepl.run_translate(
        text=gpt_response,
        from_lang='EN',
        to_lang='JA'
    )

    print(translate_result)


if __name__ == '__main__':
    main()
