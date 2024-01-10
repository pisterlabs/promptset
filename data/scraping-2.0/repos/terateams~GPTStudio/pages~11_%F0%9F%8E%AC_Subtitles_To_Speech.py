import streamlit as st
from openai import OpenAI
import xml.etree.ElementTree as ET
import srt
import re
from datetime import timedelta
import tempfile

from pydub import AudioSegment

from libs.llms import translate_srt, merge_srt
from libs.session import PageSessionState

page_state = PageSessionState("subtitles_to_speech")

page_state.initn_attr("srt_content", "")

# 设置 Streamlit
st.title("🎬 字幕语音合成")

# 上传字幕文件
uploaded_file = st.file_uploader("上传字幕文件", type=["xml", "ttml", "srt"])
col1, col2 = st.columns(2)
parse_btn = col1.button("解析字幕")
clear_btn = col2.button("清除字幕")
audio_box = st.container()

with st.sidebar:
    sound_role = st.selectbox("选择音色", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=0)
    status_bar = st.progress(0.0, text="")
    ttl_button = st.button("合成语音")

if clear_btn:
    page_state.srt_content = ""
    st.rerun()


# 函数：创建静音音频
@st.cache_data
def create_silence(duration_milliseconds):
    # 生成指定时长的静音音频
    silence = AudioSegment.silent(duration=duration_milliseconds)
    return silence


def merge_overlapping_subtitles(subtitles_src):
    subtitles = [s for s in srt.parse(subtitles_src)]
    merged_subtitles = []
    buffer_sub = subtitles[0]

    for sub in subtitles[1:]:
        if sub.start != buffer_sub.end:
            # 合并字幕
            buffer_sub.content += sub.content
            # buffer_sub.end = max(buffer_sub.end, sub.end)
        else:
            # 保存并开始新的字幕段
            merged_subtitles.append(buffer_sub)
            buffer_sub = sub

    # 添加最后一个字幕段
    merged_subtitles.append(buffer_sub)
    return srt.compose(merged_subtitles)


def ttml_to_srt(ttml_content):
    # 解析 TTML 内容
    root = ET.fromstring(ttml_content)

    # 提取 <p> 元素，这些是字幕段
    subtitles = root.findall('.//{http://www.w3.org/ns/ttml}p')

    # SRT 格式的结果
    srt_result = ""

    for index, subtitle in enumerate(subtitles):
        # 提取开始和结束时间
        begin = subtitle.get('begin')
        end = subtitle.get('end')

        # 转换时间格式为 SRT 格式（HH:MM:SS,MMM）
        begin_srt = re.sub(r'(\d{2}):(\d{2}):(\d{2}).(\d{3})', r'\1:\2:\3,\4', begin)
        end_srt = re.sub(r'(\d{2}):(\d{2}):(\d{2}).(\d{3})', r'\1:\2:\3,\4', end)

        # 提取字幕文本
        text = ''.join(subtitle.itertext())

        # 组装 SRT 字幕段
        srt_result += f"{index + 1}\n{begin_srt} --> {end_srt}\n{text}\n\n"

    return srt_result


# 函数：转换 XML 字幕到 SRT 格式
def xml_to_srt(xml_content):
    root = ET.fromstring(xml_content)
    subtitles = []

    for p in root.findall('.//p'):
        # 获取开始时间和持续时间
        start_time = int(p.get('t', 0))
        duration = int(p.get('d', 0))
        end_time = start_time + duration

        # 创建 SRT 字幕段
        subtitle_segment = srt.Subtitle(
            index=len(subtitles) + 1,
            start=timedelta(milliseconds=start_time),
            end=timedelta(milliseconds=end_time),
            content=''.join(s.text for s in p.findall('.//s') if s.text)
        )
        subtitles.append(subtitle_segment)

    return srt.compose(subtitles)


# 函数：生成语音文件并返回 AudioSegment 对象
@st.cache_data
def generate_speech_segment(text):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice=sound_role,
        input=text
    )
    # 保存到临时文件并读取为 AudioSegment
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        response.stream_to_file(temp_file.name)
        return AudioSegment.from_mp3(temp_file.name)


# 函数：处理 SRT 并生成音频片段
def process_srt_and_generate_audio(srt_content):
    segments = []
    previous_end_time = timedelta(0)

    counter = 0
    data = [s for s in srt.parse(srt_content)]
    for subtitle in data:
        if not subtitle.content.strip():
            segments.append(create_silence((subtitle.end - subtitle.start).total_seconds() * 1000))
            continue
        # 处理字幕间的空白时间
        silence_duration_ms = int((subtitle.start - previous_end_time).total_seconds() * 1000)
        if silence_duration_ms > 0:
            segments.append(create_silence(silence_duration_ms))

        # 生成语音段
        speech_segment = generate_speech_segment(subtitle.content)
        segments.append(speech_segment)

        # 更新前一个字幕的结束时间
        previous_end_time = subtitle.end
        counter += round(1 / len(data), 1)
        status_bar.progress(counter, text=f"合并音频文件 {counter * 100:.2f}%")

    return segments


# 合并所有音频文件
# 函数：合并音频片段
def merge_audio_segments(segments):
    combined = AudioSegment.empty()
    for segment in segments:
        combined += segment
    return combined


if parse_btn:
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'xml':
            # 转换 XML 到 SRT
            page_state.srt_content = xml_to_srt(uploaded_file.getvalue().decode())
        if file_type == 'ttml':
            # 转换 XML 到 SRT
            page_state.srt_content = ttml_to_srt(uploaded_file.getvalue().decode())
        elif file_type == 'srt':
            # 直接读取 SRT 文件
            page_state.srt_content = uploaded_file.getvalue().decode()
        else:
            st.error("不支持的文件格式。请上传 XML 或 SRT 文件。")
    else:
        st.error("请先上传字幕文件。")

if page_state.srt_content:
    with st.form(key='subtitles_to_speech_form'):
        srt_text = st.text_area("字幕内容，可修改", page_state.srt_content, height=480)
        form_submit_button = st.form_submit_button(label='更新字幕')
        if form_submit_button:
            ts = list(srt.parse(srt_text))
            for i in range(len(ts)):
                ts[i].index = i + 1
            page_state.srt_content = srt.compose(ts)
            st.rerun()

if ttl_button:
    status_bar.progress(0.0, text=f"开始合成音频文件")
    audio_segments = process_srt_and_generate_audio(page_state.srt_content)

    # 合并音频段
    merged_audio = merge_audio_segments(audio_segments)
    status_bar.progress(1.0, text=f"合并音频文件 100%")

    # 导出为文件
    merged_audio_path = 'final_audio.mp3'
    merged_audio.export(merged_audio_path, format='mp3')

    # 提供下载链接
    audio_box.audio(merged_audio_path)
    audio_box.download_button(label="下载音频文件",
                              data=open(merged_audio_path, 'rb'),
                              file_name="final_audio.mp3",
                              mime="audio/mp3")
