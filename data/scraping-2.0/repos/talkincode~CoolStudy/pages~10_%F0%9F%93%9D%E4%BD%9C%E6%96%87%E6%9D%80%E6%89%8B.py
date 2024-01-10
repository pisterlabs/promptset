import streamlit as st
import sys
import os
from dotenv import load_dotenv
from libs.llms import openai_streaming

sys.path.append(os.path.abspath('..'))
load_dotenv()

st.markdown("## 📝 作文杀手")
st.markdown("教你用一种新的方式写作文，让你的作文更加生动有趣，更加有逻辑。")

topic = st.text_input("输入作文题目：", "我的梦想")
remark = st.text_area("写作要求：", "我的梦想是当一名科学家。")
if st.button("开始写作"):
    with st.spinner("生成中..."):
        msg = f"""
        请按照我的要求写一篇中学生的作文。
        - 语言风格要符合中学生的特点；
        - 六百到七百字；
        - 适当引用诗词，成语，谚语；
        - 弄清用户输入题材;
        - 开头简洁，中间内容丰富，不能有废话，结尾点题，呼应标题，总结全文;
        - 内容要符合用户输入的职业；
        - 讲究格式；
        
        写作主题；{topic}
        写作要求：{remark}
        作文内容："""
        response = openai_streaming(msg,[])
        placeholder = st.empty()
        full_response = ''
        for item in response:
            text = item.content
            if text is not None:
                full_response += text
                placeholder.markdown(full_response)
        placeholder.markdown(full_response)

