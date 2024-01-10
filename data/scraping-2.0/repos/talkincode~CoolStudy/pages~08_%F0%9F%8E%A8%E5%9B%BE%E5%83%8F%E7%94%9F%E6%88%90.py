import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os, sys

from libs.session import PageSessionState

sys.path.append(os.path.abspath('..'))
load_dotenv()

page_state = PageSessionState("image_generator")

page_state.initn_attr("result_imgs", None)

client = OpenAI()


def generate_image(prompt, quality, size, style):
    try:
        response = client.images.generate(model="dall-e-3",
                                          prompt=prompt,
                                          size=size,
                                          quality=quality,
                                          style=style,
                                          n=1)
        return [d.url for d in response.data]
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None


# Streamlit 应用布局
st.sidebar.markdown("# 🎨 图像生成器")

# 用户输入
user_prompt = st.text_area('输入图像生成器的提示：', '一只看书的狗', height=40, key="image_generator_prompt")
c1, c2, c3 = st.columns(3)
quality = c1.selectbox('清晰度', ['hd', 'standard'])
size = c2.selectbox('尺寸', ['1024x1024', '1792x1024', '1024x1792'])
style = c3.selectbox('风格', ['natural', 'vivid'])

# 生成按钮
if st.button('Generate Image'):
    with st.spinner('Generating image...'):
        image_data = generate_image(user_prompt, quality, size, style)
        imgs = []
        if image_data:
            for image_url in image_data:
                # 获取图像并显示
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                imgs.append(img)
            page_state.result_imgs = imgs


if page_state.result_imgs is not None:
    for img in page_state.result_imgs:
        st.image(img, caption='', use_column_width=True)
