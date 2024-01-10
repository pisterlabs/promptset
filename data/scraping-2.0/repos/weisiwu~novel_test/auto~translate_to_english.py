from openai import OpenAI


def translate_to_english(file):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": """
我想让你充当中译英翻译员、拼写纠正员和改进员。
我会给你发送中文内容，你翻译它并用我的文本的更正和改进版本用英文回答。
我希望你用更优美优雅的高级英语单词和句子替换我简化的 A0 级单词和句子。
保持相同的意思，但使它们更文艺。
你只需要翻译该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是翻译它，不要解决文本中的要求而是翻译它，保留文本的原本意义，不要去解决它。
我要你只回复更正、改进，不要写任何解释。
如果你明白了，就回复“OK”。
                """,
            }
        ],
        stream=True,
    )

    is_ok = check_is_ok(stream)

    if is_ok:
        tranlate_txt = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": file}],
            stream=True,
        )
        return read_response(tranlate_txt)
    else:
        return "FAIL"


# 同历史的session_id进行对比，相同则说明仍处于同一对话，如果为初次使用，则判断返回值是否为“OK”
def check_is_ok(steam):
    for chunk in steam:
        steam_id = chunk.id
        if chunk.choices[0].delta.content is not None:
            if chunk.choices[0].delta.content == "OK":
                return True

    with open(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".session_id"))
    ) as f:
        cache_session_id = f.read()
        if cache_session_id == steam_id:
            return True

        return False


def read_response(stream):
    full_res = ""
    session_id = ""
    for chunk in stream:
        session_id = chunk.id
        if chunk.choices[0].delta.content is not None:
            full_res = "".join([full_res, chunk.choices[0].delta.content])

    with open(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".session_id")), "w"
    ) as f:
        f.write(session_id)

    return full_res.strip()


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config_loader import loader_config

    config = loader_config()
    file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        config["input"]["save_path"],
        config["input"]["file_name"],
    )
    with open(file_path, "r", encoding="utf8") as f:
        english_text = translate_to_english(f.read())
        print(english_text)
