from typing import List, Union

import pandas as pd

from st_aggrid import AgGrid, GridOptionsBuilder

import streamlit as st
import diskcache
from oai_client import OAIClient
from settings import Settings
import utils
import av
import numpy as np
import streamlit_webrtc as webrtc
from audio_recorder_streamlit import audio_recorder
import streamlit_webrtc as webrtc
import speech_recognition as sr
import tempfile 
import os
import azure.cognitiveservices.speech as speechsdk
import openai
from io import BytesIO
import io





# MODELS = [
#     "text-davinci-003",
#     "text-davinci-002",
#     "text-curie-001",
#     "text-babbage-001",
#     "text-ada-001",
#     "code-davinci-002",
#     "code-cushman-001",
# ]
positionType = [
    "前端工程师",
    "后端工程师",
    "计算机视觉工程师",
    "NLP算法工程师",
    "测试开发工程师",
    "人力资源专员hr",
    "会计",
]




# 设置服务密钥和终结点
speech_key = "8046cb11ab7a494da541e0187e1a1c2d"
service_region = "eastus"


# 创建一个SpeechConfig对象并设置密钥和区域
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# 设置语音合成的语言和语音样式（根据需要进行修改）
speech_config.speech_synthesis_language = "zh-CN"
speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
# 创建一个SpeechSynthesizer对象
# file_name = "audiofile.wav"
# file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
    

# synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# result = synthesizer.speak_text_async(text).get()


def convert_speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    
    # 创建一个临时文件，将音频数据写入其中
    temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
    with open(temp_audio_fd, 'wb') as temp_audio:
        temp_audio.write(audio_bytes)

    # 使用临时文件进行语音识别
    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio, language='zh-CN')  # 使用Google语音识别引擎，语言为中文
        return text
    except sr.UnknownValueError:
        print("语音识别无法理解")
    except sr.RequestError:
        print("无法连接到语音识别服务")
    
    # 删除临时文件
    os.remove(temp_audio_path)
    
    return None
# STOP_SEQUENCES = [
#     "Candidate:",
#     "Interviewer:",
#     "newline",
#     "double-newline",
#     "Human:",
#     "Assistant:",
#     "Q:",
#     "A:",
#     "INPUT",
#     "OUTPUT",
# ]
FEEDBACK_PROMPT = """
(面试结束)
请就候选人在面试中的表现提供反馈。 即使他们的简历很棒
关注他们的面试表现很重要。 如果聊天时间很短而你还不够
提供反馈信息，请提供简历反馈。 并解释你
希望在面试中看到更多的候选人。
通过面试的表现、候选人的简历信息给出一些岗位匹配推荐
请包括以下信息：
* 候选人优势
* 候选人的弱点
* 总体结论
* 雇用/不雇用建议
* 推荐岗位

您的反馈应采用以下格式：

优势：

<在此列出优势>

弱点：

<在此列出弱点>

结论：

<在此列出的结论>

建议：<雇用/不雇用>
如果建议不雇用，推荐岗位选择输出无

推荐岗位：
<在此列出的推荐岗位>

您的反馈：
""".strip()

INITIAL_TRANSCRIPT = "Interviewer: 你好"

INITIAL_RESUME = """
工作经历：
京东
2022年07月 - 至今
前端工程师
北京
负责京东金融跨端技术开发，适配android、ios、h5实现一码三端，综合了web生态和native组件，让js执行代码后用 native的组件进行渲染提高研发效率
通过h5容器实现原生页面与h5的交互，实现站外快速唤起以及业务解耦，提供基础支持能力
基于自研跨端方案为更多业务赋能，使用跨端方案人均吞吐量提升了80%以上
项目经历：
京东金融
共建跨端技术底层sdk，为业务提供基础支撑能力
增加组件库市场，对业务中的用到的模块组件化，便于代码复用提高开发效率
通过h5容器对应用的权限管控、合规检测、js桥接native以及h5生态

技能：vuejs、javascript；python、java开发；熟悉常用的深度学习算法，目标检测及跟踪；
语言：英语（CET-6）

"""

INITIAL_QUESTION = """
You are an professional interviewer in a tech company. You're interviewing the user who applied for {{position}}. 


CANDIDATE RESUME:

{{resume}}
(END OF RESUME)

INTENDED POSITION:

{{position}}

(END OF POSITION)

The interview should adhere to the following format:

2mins- opening intros ，"tell me about yourself" questions
10mins - ask the candidate experience which is relevant to the {{position}} in details, especially using "how"questions
5mins - ask the candidate professional knowledge details relevant to the {{position}}
2mins - ask the candidate if they have any questions for you
thanks the candidate for his time, tell when the decision will be made, and the interview ends

Some clarifications (if the candidate asks or it feels appropriate to share):

1. What are the expected response time and throughput of this service?

Ideally within 1 second each time the user changes their query or types a new word or words.

2. How many suggestions need to be displayed in response to a query?

5 to 10 suggestions

3. If the user inputs "stop", "bye" this kind of words, terminate the interview.
4. Do not repeat the interview.

Here are the rules for the conversation:
* You are a chat bot who conducts system design interviews
* Speak in first person and converse directly with the candidate
* Do not provide any backend context or narration. Remember this is a dialogue
* Do NOT write the candidates's replies, only your own
* 提问的方式最好由浅入深，同时不要重复问一个问题，当时候选人回答不会、能换个问题么的时候换一个面试问题
* We don't have access to a whiteboard, so the candidate can't draw anything. Only type/talk.
* 用中文回答
BEGIN!

{{transcript}}
""".strip()


@st.cache(ttl=60 * 60 * 24)
def init_oai_client(oai_api_key: str):
    cache = diskcache.Cache(directory="/tmp/cache")
    oai_client = OAIClient(
        api_key=oai_api_key,
        organization_id=None,
        cache=cache,
    )
    return oai_client


def run_completion(
    oai_client: OAIClient,
    prompt_text: str,
    model: str,
    stop: Union[List[str], None],
    max_tokens: int,
    temperature: float,
    best_of: int = 1,
):
    print("Running completion!")
    if stop:
        if "double-newline" in stop:
            stop.remove("double-newline")
            stop.append("\n\n")
        if "newline" in stop:
            stop.remove("newline")
            stop.append("\n")
    resp = oai_client.complete(
        prompt_text,
        model=model,  # type: ignore
        max_tokens=2048,  # type: ignore
        temperature=temperature,
        stop=stop or None,
        best_of=best_of,
    )
    return resp


def get_oai_key():
    import os
    oai_key = os.environ.get("OPENAI_API_KEY")
    if oai_key is None:
        raise Exception("Must set `OPENAI_API_KEY` environment variable or in .streamlit/secrets.toml")
    return oai_key



def main():
    utils.init_page_layout()
    session = st.session_state
    oai_client = init_oai_client(get_oai_key())
    openai.api_key = get_oai_key()
    if "transcript" not in session:
        session.transcript = [INITIAL_TRANSCRIPT]
        session.candidate_text = ""
        session.resume_text = ""
        session.custom_prompt_text = INITIAL_QUESTION

    

    # with st.sidebar:
    #     max_tokens = st.number_input(
    #         "Max tokens",
    #         value=2048,
    #         min_value=0,
    #         max_value=2048,
    #         step=2,
    #     )
    #     temperature = st.number_input(
    #         "Temperature", value=0.7, step=0.05
    #     )
    stop = ["Candidate:", "Interviewer:"]

    resume_tab, chat_tab,feedback_tab = st.tabs(["简历填写", "面试", "面试反馈"])

    with resume_tab:
        # st.markdown("### 上传简历")
        resume = st.file_uploader("上传简历 ☁", type = 'pdf')
        # pdf_reader = PdfReader(self.pdf_file)
        #     text = ""
        def clear_ResumeText():
            session.transcript.append(f" {resume_text.strip()}")
            session["resume_text"] = ""
        # resume_text = resume_tab.text_area(
        #     "候选人简历",
        #     height=500,
        #     key="resume_text",
            
        # )
        resume_text = INITIAL_RESUME
        position = st.selectbox(
            "岗位",
            positionType,
           
        )
        print("**********选择岗位是*************\n",resume_text)
    # run_button1 utton("提交",on_click=clear_ResumeText)
    
    # with question_tab:
    #     question_text1 = question_tab.text_area(
    #         "Question Prompt",
    #         height=700,
    #         key="custom_prompt_text",
    #         value=INITIAL_QUESTION,
    #     )
    # with question_tab2:
    #     question_text2 = question_tab2.text_area(
    #         "自定义Prompt",
    #         height=700,
    #         key="custom_prompt_text",
    #     )
        question_text1 = INITIAL_QUESTION
        print("========初始prompt=========\n", question_text1)
        print("*******************************************\n\n")
        

        


    with chat_tab:
        st.write("\n\n".join(session.transcript))


        def clear_text():
            session.transcript.append(f"Candidate: {candidate_text.strip()}")
            session["candidate_text"] = ""
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="1x",
        )
        if audio_bytes:
            st.session_state.audio_bytes = audio_bytes
            audio_file = io.BytesIO(st.session_state.audio_bytes)
            audio_file.name = "temp_audio_file.wav"
            text = openai.Audio.transcribe("whisper-1", audio_file)
            text = text['text']
            candidate_text = chat_tab.text_area(
                "Interview Chat",
                height=50,
                # key="candidate_text",
                help="Write the candidate text here",
                value = text
            )
        else:
            candidate_text = chat_tab.text_area(
                "Interview Chat",
                height=50,
                key="candidate_text",
                help="Write the candidate text here",

            )


        run_button = st.button("Enter", help="Submit your chat", on_click=clear_text)


        if run_button:
            if not resume_text:
                st.error("请输入简历")
            if not question_text1:
                st.error("Please enter a question")

            def choosePrompt(question_text1, question_text2):
                if not question_text2:
                    question_text = question_text2
                else:
                    question_text = question_text1
                print("question_text", question_text)
                return question_text
            prompt_text = utils.inject_inputs(
                question_text1, input_keys=["transcript", "resume"], inputs={
                    "transcript": session.transcript,
                    "resume": resume_text,
                    "position":position,
                }
            ) + "\nInterviewer:"
            print("prompt_text\n\n", prompt_text)

            resp = run_completion(
                oai_client=oai_client,
                prompt_text=prompt_text,
                model="text-davinci-003",  # type: ignore
                stop=stop,
                max_tokens=2048,  # type: ignore
                temperature=0.5,
            )
            completion_text = resp["completion"].strip()
            if completion_text:
                print("Completion Result: \n\n", completion_text)
                # speak_text(completion_text)
            #文本显示面试官的回答
            session.transcript.append(f"Interviewer: {completion_text}")
            st.experimental_rerun()
                #语音读出面试官的回答
                # synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
                # result = synthesizer.speak_text_async(completion_text).get()

        with feedback_tab:
            st.header("候选人面试反馈")
            def choosePrompt(question_text1, question_text2):
                if not question_text2:
                    question_text = question_text2
                else:
                    question_text = question_text1
                print("question_text", question_text)
                return question_text
            prompt_text = utils.inject_inputs(
                question_text1, input_keys=["transcript", "resume"], inputs={
                    "transcript": session.transcript,
                    "resume": resume_text,
                    "position":position,
                }
            )
            feedback_prompt_text = prompt_text + "\n\n" + FEEDBACK_PROMPT
            if st.button("生成面试反馈"):
                resp = run_completion(
                    oai_client=oai_client,
                    prompt_text=feedback_prompt_text,
                    model="text-davinci-003",  # type: ignore
                    stop=stop,
                    max_tokens=2048,  # type: ignore
                    temperature=0.5,
                )
                st.write(resp["completion"])

                

if __name__ == "__main__":
    main()
