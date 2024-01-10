import streamlit as st
import os
import requests
from openai import OpenAI

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

input_text = st.text_input("生成したい画像を指示してください")
create_num = st.number_input("生成画像の枚数", value=1, step=1)

if st.button("実行"):
    st.write(input_text)

    from openai import OpenAI

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "あなたはプロの翻訳家です。次の文章を英語に翻訳してください。"},
            {"role": "user", "content": f"{input_text}"}
        ]
    )
    eng_prompt = response.choices[0].message.content

    image = client.images.generate(
        model="dall-e-2",
        prompt=eng_prompt,
        size="256x256",
        quality="standard",
        n=create_num,
    )

    for data in image.data:
        image_url = data.url
        st.image(image_url)

        # 画像をダウンロードするための一時ファイルを作成
        response = requests.get(image_url)
        if response.status_code == 200:
            with open("temp_image.png", "wb") as file:
                file.write(response.content)

            # ダウンロードボタンを追加
            with open("temp_image.png", "rb") as file:
                st.download_button(
                    label="画像をダウンロード",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
