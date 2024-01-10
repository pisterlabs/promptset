import os
import json

import requests
from langchain.document_loaders import UnstructuredFileLoader
import streamlit as st
import openai
import random

from learnBi.mycomponent import mycomponent


def make_card(text):
    openai.api_key = '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'
    # text = "仅仅通过神经元进行存储数据，存储能力十分有限。NTM[12]最早提出了外部记忆增强神经网络架构，通过一个大的可寻址的存储器来扩展存储的能力，实现存储管理并且可微。神经图灵机的灵感来自于图灵机的架构，由控制器、并行读写头和外部存储器组成，将神经网络和外部存储器结合来扩展神经网络的能力，可以使用梯度下降进行高效训练。NTM 可以通过选择性的读写操作与内存矩阵进行交互。可以通过两种方式进行寻址，一种是基于内容的寻址，另外一种是基于位置的寻址。"
    # prompt = "请根据我提供的文本，生成一套抽认卡。在制作抽认卡的时候，请循环下述要求：1.保持抽认卡的简单、清晰，并集中于最重要的信息2.确保答案是具体的，使用简单清晰的语言3.尊重事实，并使卡片便于阅读和理解4.使用与原文本相同的语言进行回答"
    prompt = "请根据我提供的文本，制作一套抽认卡。\
            在制作抽认卡时，请遵循下述要求：\
            1. 保持抽认卡的简单、清晰，并集中于最重要的信息。\
            2. 确保问题是具体的、不含糊的。\
            3. 使用清晰和简洁的语言，使卡片易于阅读和理解。\
            4. 答案遵循客观事实。\
            制作抽认卡时，让我们一步一步来：\
            第一步，结合上下文，使用简单的相同语言改写内容，同时保留其原来的意思。\
            第二步，将内容分成几个小节，每个小节专注于一个要点。\
            第三步，利用小节来生成多张抽认卡，对于超过50个字的小节，先进行拆分和概括，再制作抽认卡。只生成最重要的内容即可。\
            文本：衰老细胞的特征是细胞内的水分减少，结果使细胞萎缩，体积变小，细胞代谢的速率减慢。细胞内多种酶的活性降低。细胞核的体积增大，核膜内折，染色质收缩、染色加深。细胞膜通透性改变，使物质运输功能降低。\
            一套卡片：\n\
    卡片1：\n\
    问题：衰老细胞的体积会怎么变化？\n\
    答案：变小。\n\
    卡片2：\n\
    问题：衰老细胞的体积变化的具体表现是什么？\n\
    答案：细胞萎缩。\n\
    卡片3：\n\
    问题：衰老细胞的体积变化原因是什么？\n\
    答案：细胞内的水分减少。\n\
    卡片4：\n\
    问题：衰老细胞内的水分变化对细胞代谢的影响是什么？\n\
    答案：细胞代谢的速率减慢。\n\
    卡片5：\n\
    问题：衰老细胞内的酶活性如何变化？\n\
    答案：活性降低。\n\
    卡片6：\n\
    问题：衰老细胞的细胞核体积如何变化？\n\
    答案：体积变大。\n\
    卡片7：\n\
    问题：衰老细胞的细胞核的核膜如何变化？\
    答案：核膜内折。\
    卡片8：\
    问题：衰老细胞的细胞核的染色质如何变化？\n\
    答案：染色质收缩。\n\
    卡片9：\n\
    问题：衰老细胞的细胞核的染色质变化对细胞核形态的影响是？\n\
    答案：染色加深。\n\
    卡片10：\n\
    问题：衰老细胞的物质运输功能如何变化?\n\
    答案：物质运输功能降低。\n\
    卡片11：\n\
    问题：衰老细胞的物质运输功能为何变化？\n\
    答案：细胞膜通透性改变。\n\
            文本："

    def chat_with_gpt(p):

        url = "https://openai.api2d.net/v1/completions"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer fk205005-4JjeuMSr5qUREGOdRyqpS0pWQ6iAf6sM'
            # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
        }

        data = {
            "model": "text-davinci-003",
            "prompt": p,
            "max_tokens": 2000,
            "n": 5,
            "stop": None,
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['text'].strip()

    response = chat_with_gpt(prompt + text + "\n一套卡片：\n")
    print(response)

    def colorful_card(title, ques, ans, color):
        style = f"""
            background-color: {color};
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            width: 400px; 
            height: 260px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        """
        container_style = """
                display: flex;
                flex-direction: column;
                align-items: center;
            """
        content = f"{ques}<br>{ans}"
        card_html = f"""
            <div style="{container_style}">
                <div style="{style}">
                    <h2>{title}</h2>
                    <p>{content}</p>
                </div>
            </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    titles = []
    ques = []
    ans = []
    colors = ["#98FF98", "#FFC0CB", "#C8A2C8", "#87CEEB", "#FFFACD", "#ADD8E6", "#32CD32", "#E6E6FA", "#00CED1",
              "#90EE90", "#FFD700"]
    lines = response.splitlines()
    lines = [s for s in lines if s != '']
    print(lines)
    random_elements = random.sample(colors, len(lines) // 3)
    print(random_elements)
    for i in range(len(response.splitlines())):
        if i % 3 == 0:
            titles.append(lines[i])
        if i % 3 == 1:
            ques.append(lines[i])
        if i % 3 == 2:
            ans.append(lines[i])
    print(titles)
    print(ques)
    print(ans)

    for i in range(len(ans)):
        colorful_card(titles[i], ques[i], ans[i], random_elements[i])


def get_first_card(text):
    messages = [{"role": "user", "content": f'''Imagine you are a Text-to-Card Converter. Your task is to take lengthy pieces of text and break them down into several small, easily digestible cards for the user to read. Each card should encapsulate a focused concept but also need to faithfully replicate the original text, including a title and content. Importantly, the language used in the cards must be in Chinese. Some parts may have formatting issues, please fix them. Below is the original text.
    ---------------------------------
    {text}'''}]
    functions = [
        {
            "name": "get_first_card",
            "description": "Get first card in a given text",
            "parameters": {
                "type": "object",
                "properties": {
                    "card": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title, e.g. Concept of RLHF, keep it blank if not focused enough",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content",
                            },
                        }
                    },
                    "remaining": {
                        "type": "string",
                        "description": "The first 10 words of remaining text that is not included in the first card",
                    },
                },
                "required": ["card", "remaining"],
            },
        }
    ]
    import requests

    url = "https://openai.api2d.net/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer fk205005-4JjeuMSr5qUREGOdRyqpS0pWQ6iAf6sM'
        # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }

    data = {
        "model": "gpt-3.5-turbo-0613",
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code", response.status_code)
    print("JSON Response ", response.json())
    return response.json()

st.header("PDF Import and Display")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'docx', 'txt'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    with open(os.path.join("pdf_files", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = os.path.join("pdf_files", uploaded_file.name)
    st.write(file_path)
    loader = UnstructuredFileLoader(file_path, mode="elements")
    docs = loader.load()
    print([doc.page_content for doc in docs])
    text = '\n'.join([doc.page_content for doc in docs])
    print(st.session_state)

    selected = None
    if 'cards' in st.session_state:
        for card in st.session_state.cards:
            value = mycomponent(my_input_value=f'<em>{card["title"]}</em><br>{card["content"]}')
            if value and len(value) > 0:
                selected = value

    if selected:
        st.write(selected)
        make_card(selected)



    if 'remaining' not in st.session_state or len(st.session_state.remaining) > 10:
        if st.button('继续'):
            if 'remaining' not in st.session_state:
                st.session_state.remaining = text
                st.session_state.cards = []

            arguments = json.loads(get_first_card(st.session_state.remaining[:1000])['choices'][0]['message']['function_call']['arguments'])
            st.session_state.remaining = st.session_state.remaining[st.session_state.remaining.find(arguments["remaining"][:4]):]
            st.session_state.cards.append(arguments["card"])
            st.experimental_rerun()