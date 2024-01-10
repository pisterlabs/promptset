from urllib import response
import streamlit as st
import re
import datetime
from dotenv import load_dotenv
import os
import openai
 
load_dotenv('config.env') 
 
st.set_page_config(layout='wide')
st.title('Deciphr Transcript Curation and QnA Demo')

FILENAME = "INT-072622-112855 _ Test Audio File.json"
AUDIO_URL = "https://storage.googleapis.com/tektorch_podcast_audio/INT-072622-112855_Test_Audio_File.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=hyfen8%40appspot.gserviceaccount.com%2F20220812%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220812T132117Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=88beb7baf258a07841c5f943cc20c49f88a02ca3615e4f94ebfc1d2261abd3505d3e93008981b4bbe2c5e69efca504e6c585a6f4337aeeb5e36f37381be7612708d91e3987cba6395b15d3407847f464655cb6bab9581bec7551050971b612bc41b67618e945506e033c3fa684b3ffcf6121149edf52e7e4691d2e28a4fd4e26898fb5daca3a99efb50a86f182642cf61b80b2643564d813547014f93f6ef714906b595e028d3c4fa1b35c1001b79e871e8eef8166b36e3c6462c22322dc6878cdbd9c83726d416ad75e3fd9c84d42f496071b27e79422a745ad9f90ec2765c3429f476525cea41ee4293edb911280ea30cfe5b39c6eb80ef8a7ca2e853385ca"
RAW_INFO = """INFO: Expand to show full transcript."""
CONF_INFO = """INFO: Expand to show highlighted transcript data.\n
Highlighted text has low a confidence value and may need manual correction.\n
Press Play to start playing audio from the timestamp displayed."""

THICK_LINE = '<hr style="height:5px;background-color: #ffffff"/>'
st.markdown(THICK_LINE, unsafe_allow_html=True)

OPENAI_KEY = os.environ.get("OPENAI_KEY")
openai.api_key = OPENAI_KEY

if 'raw_transcript' not in st.session_state:
    st.session_state['raw_transcript'] = ""

if 'start_time' not in st.session_state:
    st.session_state['start_time'] = '00:00:00'

if "conf_data" not in st.session_state:
    st.session_state['conf_data'] = []

if "answer" not in st.session_state:
    st.session_state['answer'] = ""

if 'transcript_data' not in st.session_state:
    # read json file
    import json
    with open('INT-072622-112855 _ Test Audio File.json', 'r') as f:
        transcript_data = json.load(f)
    st.session_state['transcript_data'] = transcript_data

def transcript_formatter(data):
    try:
        utterences = data['utterances']
        out = ""
        for utterence in utterences:
            timestamp = utterence['start']
            ts = str(datetime.timedelta(seconds=int(timestamp/1000)))
            speaker = utterence['speaker']
            text = utterence['text']
            str_f = '{}: [{}] {}\n\n'.format(speaker, ts,text)
            out += str_f
        return out
    except:
        return None

def get_conf_data():
    conf_list = []
    for u in st.session_state['transcript_data']['utterances']:
        ts_raw = u['start']
        ts = str(datetime.timedelta(seconds=int(ts_raw/1000)))
        speaker = u['speaker']
        word_conf = {}
        word_conf['ts'] = ts
        word_conf['ts_raw'] = ts_raw
        word_conf['speaker'] = speaker
        word_conf_list = []
        for i in u['words']:
            if i['confidence'] > 0.5:
                word_conf_list.append([i['text'],1])
            else:
                word_conf_list.append([i['text'],0])
        word_conf['words'] = word_conf_list
        conf_list.append(word_conf)
    return conf_list

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def convert_to_html(l):
    html = ""
    for i in l:
        word = i[0]
        if i[1] == 0:
            html += "<span class='highlight blue'>{} </span>".format(word)
        else:
            html += "{} ".format(word)
    html = "<div>{}</div>".format(html)
    return html

def audio_player_html(ts=st.session_state['start_time']):
    if len(ts) == 7:
        ts = "0"+ts
    if ts=="00:00:00":
        ap = ""
    else:
        ap = 'autoplay'
    html = """
    <div style="padding-top: 13px; right: 0px; display: flex; justify-content: center; align-items: center; background-color:#1c1c1c; width: 100%; position: fixed; top:86vh; z-index:2;">
    <div>
    <h6>{}</h6>
    </div>
    </div> 
    <div style="padding-bottom: 30px;padding-top: 20px; right: 0px; display: flex; justify-content: center; align-items: center; background-color:#1c1c1c; width: 100%; position: fixed; top:90vh; z-index:2;">
    <audio style="width:70%"
        controls {}
        src="{}#t={}">
            Your browser does not support the
            <code>audio</code> element.
    </audio>
    </div>
    """.format(FILENAME,ap,AUDIO_URL, ts)
    return html

def set_start_time(ts):
    st.session_state['start_time'] = ts

def submit_qna():
    if st.session_state['qna_input']:
        documents = [u['text'] for u in st.session_state['transcript_data']['utterances']]
        question = st.session_state['qna_input']
        response = openai.Answer.create(
            search_model="davinci",
            model="davinci",
            question=question,
            documents=documents,
            examples_context="Yeah, I think the market in the US is undervalued for radiation oncology. If you look at treatment patterns in European countries that have more state supported insurance reimbursement, they utilize radiation treatments much more readily and actually invest more in the research of radiation treatment than high risk, high reward individual targeted oncology therapies where in the United States, the research model is driven primarily by chemo companies, both in large company biotech as well as small startups to individualized medicine. We saw a bleed over I saw a bleed over of individualized medicine trying to understand individual response to radiation therapy and so much more reliance on genetics and tumor profile to understand the best treatment course. And I think from a market perspective. As the United States moves towards potentially more federal supported widening of Medicare and Medicaid. We will see an increase in folks that are directed to radiation treatment as opposed to high cost individualized therapies. Especially when we start talking about rural America and where access to state of the art care is limited due to kind of hospitalized structure throughout the United States.",
            examples=[["Describe the market for radiation oncology in the US","The market in the US is undervalued for radiation oncology. The treatment patterns in European countries that have more state supported insurance reimbursement, they utilize radiation treatments much more readily and actually invest more in the research of radiation treatment than high risk, high reward individual targeted oncology therapies. In the United States, the research model is driven primarily by chemo companies, both in large company biotech as well as small startups to individualized medicine."],["How well do we understand which patients will respond best to radiation therapy versus something else?","It depends on the type of cancer we're talking about. Certain cancers are good. For example, breast cancer, which has been researched to a point where they've been able to fine tune when and how radiation treatment is best utilized in conjunction with chemotherapy and surgery."]],
            max_tokens=64,
            stop=["\n", "<|endoftext|>"],
            )
        st.session_state['answer'] = response['answers'][0]
    else:
        return

if __name__ == "__main__":
    local_css("style.css")
    if not st.session_state['conf_data']:
        st.session_state['conf_data'] = get_conf_data()
    if not st.session_state['raw_transcript']:
        st.session_state['raw_transcript'] = transcript_formatter(st.session_state['transcript_data'])
    st.header("Deciphr Transcript Curation")
    st.subheader('Raw Transcript')
    st.info(RAW_INFO)
    with st.expander('Transcript'):
        cols = st.columns([1,1,8])
        cols[0].subheader('Speaker')
        cols[1].subheader('Timestamp')
        cols[2].subheader('Text')

        for line in st.session_state['raw_transcript'].split("\n\n"):
            try:
                speaker = line.split(':')[0]
                ts = re.findall(r'\d+:\d+:\d+', line)[0]
                text = re.findall(r'(?<=\]\s).*', line)[0]
            except:
                break
            with st.container():
                cols_ = st.columns([1,1,8])
                cols_[0].write(speaker)
                cols_[1].write(ts)
                cols_[2].write(text)
                cols_[2].write("---")

    # st.audio(data=AUDIO_URL)
    audio_html = audio_player_html()
    st.markdown(audio_html, unsafe_allow_html=True)
    st.write('---')
    st.subheader('Highlighted Transcript')
    st.info(CONF_INFO)

    with st.expander('Transcript'):
        with st.container():
            cols = st.columns([1,1,8])
            cols[0].subheader('Speaker')
            cols[1].subheader('Timestamp')
            cols[2].subheader('Text')
        for count,i in enumerate(st.session_state['conf_data']):
            with st.container():
                cols = st.columns([1,1,8])
                html = convert_to_html(i['words'])
                audio_html = audio_player_html(i['ts'])
                speaker = i['speaker']
                ts = i['ts']
                cols[0].write(speaker)
                cols[1].write(ts)
                cols[2].markdown(html, unsafe_allow_html=True)
                # cols[2].markdown(audio_html, unsafe_allow_html=True)
                cols[2].write("---")
                cols[2].button("Play", key=count, on_click=set_start_time, args=(ts,))
            st.write("---")
    st.markdown(THICK_LINE, unsafe_allow_html=True)
    st.header("Deciphr QnA")
    st.info("Ask questions about the transcript here.")
    st.text_input("Enter your question here.", key="qna_input")
    st.button("Submit", key="qna_submit", on_click=submit_qna)
    st.write('---')
    if st.session_state['answer']:
        st.subheader('Answer')
        st.success(st.session_state['answer'])
        st.markdown(THICK_LINE, unsafe_allow_html=True)


    






    