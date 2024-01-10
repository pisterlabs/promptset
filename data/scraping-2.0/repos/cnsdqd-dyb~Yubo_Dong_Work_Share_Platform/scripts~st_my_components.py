import time
import scripts.st_temp_scripts as stt
import streamlit as st
import openai
import scripts.st_better_img as stimg
def top_line():
    tab1, tab2, tab3 = st.tabs(["🎂我的主页", "🍰我的研究", "🍭我的应用"])

def my_cv():
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            stimg.render_svg('src/svg/icon.svg',width="50")
        with cols[1]:
            st.markdown("## FreedomFrank")
        st.code("https://github.com/cnsdqd-dyb")

def left_right():
    text,img = st.columns(2)
    with text:
        st.markdown("# 有朋自远方来，不亦乐乎！")
        st.caption("## Welcome my friend!")
    with img:
        #stimg.render_svg('src/svg/1876.svg')
        stimg.load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_0jQBogOQOn.json')

def self_intro():
    st.title("关于我的爱好和特长！")
    img2,text2 = st.columns(2)
    with img2:
        stimg.render_svg('src/svg/3D Guy.svg', shadow=False, width='50')
    with text2:
        with st.container():
            st.caption("## 足球爱好者")
            with st.expander("足球爱好者"):
                st.caption("more ...")
            st.caption("## 音乐爱好者")
            with st.expander("音乐爱好者"):
                st.caption("more ...")
            st.caption("## 游戏制作爱好者")
            with st.expander("游戏制作爱好者"):
                st.caption("more ...")
            st.caption("## 人工智能研究者")
            with st.expander("人工智能研究者"):
                st.caption("more ...")

class DoubleChatUI():
    def __init__(self,start_prompt="人类：你好！AI：你好！人类：接下来我们来进行一段友好的交流！AI：",key=time.time()):
        openai.api_key = st.secrets["SWEETS"].OPENAI_API_KEY
        self.start_prompt = start_prompt
        self.hash_text = str(hash(key))+'.txt'
        self.hash_textAI = str(hash(key))+'AI.txt'
        self.R = []
        self.L = []
    def read_data(self):
        self.L = stt.read(self.hash_text).split('@')
        self.R = stt.read(self.hash_textAI).split('@')
        if self.L and self.R:
            for idx in range(max(len(self.L),len(self.R))):
                if idx < len(self.L) and len(self.L[idx]) > 2:
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown('🧔:'+self.L[idx])
                if idx < len(self.R) and len(self.R[idx]) > 2:
                    c1,c2 = st.columns(2)
                    with c2:
                        st.markdown('🤖:'+self.R[idx])

    def clear_data(self):
        stt.clear(self.hash_text)
        stt.clear(self.hash_textAI)

    def chat_for(self,prompt="Create an outline for an essay about Nikola Tesla and his contributions to technology:",
                 temperature=0.9):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=3000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.6
        )
        return response['choices'][0]['text']

    def chat(self):
        self.read_data()
        text = st.text_input("🧔输入：")
        if len(text)>0:
            res = self.chat_for(prompt=text)
            st.markdown(res)
            if len(text) > 0:
                stt.add(self.hash_text, text+"@")
            if len(res) > 0:
                stt.add(self.hash_textAI, res+"@")
        del_bt = st.button('🗑删除')
        if del_bt:
            self.clear_data()
