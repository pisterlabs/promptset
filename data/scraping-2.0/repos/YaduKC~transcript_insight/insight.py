from time import sleep
from elasticsearch import Elasticsearch
import openai
import streamlit as st
import nltk
import re
import requests
import os.path

openai.api_key = st.secrets["OPENAI_KEY"]
elasticsearch_key = st.secrets["ELASTICSEARCH_KEY"]
nltk.download('stopwords')

if 'submit_' not in st.session_state:
    st.session_state.submit_ = False

if 'insight_' not in st.session_state:
    st.session_state.insight_ = []

if 'curr_tool_' not in st.session_state:
    st.session_state.curr_tool_ = ""

if 'raw_transcript_' not in st.session_state:
    st.session_state.raw_transcript_ = ""

if 'data_prep_' not in st.session_state:
    st.session_state.data_prep_ = False

if 'upload_' not in st.session_state:
    st.session_state.upload_ = False

if 'elasticsearch_data_' not in st.session_state:
    st.session_state.elasticsearch_data_ = []

if 'tiles_' not in st.session_state:
    st.session_state.tiles_ = []

if 'es_' not in st.session_state:
    st.session_state.es_ = Elasticsearch(
                    ['https://insight-08476f.es.us-east4.gcp.elastic-cloud.com'],
                    http_auth=('elastic', elasticsearch_key),
                    scheme="https", port=9243,)

def summary(chunk):
    start_sequence = "The main topic of conversation in one sentence is:"
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight
    return "Test"

def local_css():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def display_search(search_term, result, index):
    local_css()
    s = re.split(search_term.lower(), result.lower())
    html_str = "<div>"
    for count,i in enumerate(s):
        if count < len(s) - 1:
            html_str += i + "<span class='highlight blue'>"+search_term+ "</span>"
        else:
            html_str += i
    html_str = html_str+"</div>"
    with st.container():
        cols = st.columns([0.3,10])
        cols[0].write(str(index+1)+" :")
        cols[1].markdown(html_str, unsafe_allow_html=True)

    

def display_insight(data):
    with st.container():
        cols = st.columns([2,1,1])
        cols[0].header("Transcript")
        cols[1].header("Summary")
        cols[2].header("Timestamp")
    st.markdown("""---""")

    for count, i in enumerate(data):
        with st.container():
            cols = st.columns([2,1,1])
            with cols[0].container():
                with st.expander(label = "Transcript (Segment " + str(count+1) + ")"):
                    st.caption(i["transcript"])
                cols[1].caption(i["summary"])
                cols[2].caption(i["timestamp"])        
    st.markdown("""---""")


def create_tiles(transcript):
    if not st.session_state.tiles_:
        tt = nltk.tokenize.TextTilingTokenizer(w=30,k=5,smoothing_width=3, smoothing_rounds=5)
        tiles = tt.tokenize(transcript)
        st.session_state.tiles_ = tiles



def insight_generate(transcript):
    if st.button(label="Select"):
        st.markdown("""---""")
        with st.spinner("Generating Insights..."):
            if not st.session_state.insight_:
                if not st.session_state.tiles_:
                    create_tiles(transcript)
                for chunk in st.session_state.tiles_:
                    timestamps = re.findall(r"\[\d\d\:\d\d:\d\d\]", chunk)
                    if len(timestamps) >= 2:
                        timestamps = timestamps[0] + "-" + timestamps[-1]
                    elif len(timestamps) == 1:
                        timestamps = timestamps[0] + "-" + "[-:-:-]"
                    else:
                        timestamps = "[-:-:-]-[-:-:-]"
                    chunk = re.sub(r"\[\d\d\:\d\d:\d\d\]", "", chunk, flags=re.IGNORECASE)
                    insight = summary(chunk)
                    chunk_dict = {"transcript":chunk,
                                "summary":insight,
                                "timestamp":timestamps}
                    st.session_state.insight_.append(chunk_dict)
            display_insight(st.session_state.insight_)


def jsonl_converter(transcript):
    with st.spinner("Parsing Transcript..."):
        elasticsearch_data = []
        transcript_list = transcript.split("\n")
        for i in transcript_list:
            if i != "" and i[0:3] != "INT":
                i = i.strip()
                i = i.replace("\"", "")
                i = "{\"text\":" +  " \"" + i + "\"}"
                
                elasticsearch_data.append(i)
        if not st.session_state.elasticsearch_data_:
            st.session_state.elasticsearch_data_ = elasticsearch_data
        
        if os.path.isfile("data.jsonl"):
            os.remove("data.jsonl")
        file = open("data.jsonl", "w")
        if not st.session_state.tiles_:
            create_tiles(transcript)
        for tiles in st.session_state.tiles_:
            tiles = tiles.replace("\n","")
            tiles = tiles.replace("\"","")
            tiles = "{\"text\":" +  " \"" + tiles + "\"}"
            file.write(tiles + "\n")
        file.close()


        

def upload_files():
    if os.path.isfile("data.jsonl"):
        with st.spinner("Uploading Files To OpenAI Cloud..."):
            openai.File.create(file=open("data.jsonl"), purpose="search")
    else:
        st.error("Transcript JSONL File Not Found!!")

def list_curr_files():
    curr_files = []
    headers = {'Authorization': 'Bearer '+ openai.api_key}
    response_files = requests.get('https://api.openai.com/v1/files', headers=headers)
    files_metadata = response_files.json()["data"]
    for files in files_metadata:
        curr_files.append(files["id"])
    return curr_files

def delete_files():
    with st.spinner("Cleaning Workspace..."):
        headers = {'Authorization': 'Bearer '+ openai.api_key}
        response_files = requests.get('https://api.openai.com/v1/files', headers=headers)
        for i in response_files.json()["data"]:
            file_name = i["id"]
            headers = {'Authorization': 'Bearer '+ openai.api_key}
            delete_response = requests.delete('https://api.openai.com/v1/files/'+file_name, headers=headers)

    

def search():
    search_term = st.text_input(label="Enter Search Term")
    if st.button(label="Submit", key = 0):
        st.markdown("""---""")
        st.subheader("Search Results")
        res = st.session_state.es_.search(index="my-index", body={'query':{'match':{'text':search_term}}}, size=len(st.session_state.elasticsearch_data_))
        for count,hit in enumerate(res['hits']['hits']):
            display_search(search_term, hit["_source"]["text"], count)
        st.markdown("""---""")
    return None

def qna():
    #st.info("Under Construction")
    question = st.text_input(label="Enter Query")
    if st.button(label="Submit", key = 1):
        with st.spinner("Processing Answer..."):
            curr_files = list_curr_files()
            answer = openai.Answer.create(
                                search_model="davinci", 
                                model="davinci", 
                                question=question, 
                                file=curr_files[0], 
                                examples_context=st.session_state.raw_transcript_[0:2047], 
                                examples=[["What is DoubleVerify", "DoubleVerify is a digital media measurement and verification company."]],
                                max_tokens=64,
                                temperature=0.5,
                                stop=["\n", "<|endoftext|>"],
                            )
            
            st.markdown("""---""")
            st.subheader("Result")
            for i in answer["answers"]:
                st.info(i)
            st.markdown("""---""")

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return int(rightMin + (valueScaled * rightSpan))

def prepare_workspace(transcript):
    if not st.session_state.data_prep_:
        jsonl_converter(transcript)
        delete_files()
        upload_files()
        curr_files = list_curr_files()
        with st.spinner("Processing OpenAI Files..."):
            while not st.session_state.upload_:
                try:
                    output = openai.Engine("ada").search(
                                                search_model="ada", 
                                                query="Expert", 
                                                max_rerank=200,
                                                file=curr_files[0]
                                                )
                    st.session_state.upload_ = True
                except:
                    print("Processing")
                sleep(10)

        aliases = st.session_state.es_.indices.get_alias("*").keys()
        for keys in aliases:
            st.session_state.es_.indices.delete(index=keys, ignore=[400, 404])
        upload_container = st.empty()
        with upload_container.container():
            st.info("Uploading Files To ElasticCloud...")
            upload_bar = st.progress(0)
            for index, a_data in enumerate(st.session_state.elasticsearch_data_):
                percent_complete = map(index, 0, len(st.session_state.elasticsearch_data_), 0, 100) + 1
                if percent_complete <= 100:
                    upload_bar.progress(percent_complete)
                st.session_state.es_.index(index='my-index', body=a_data)
            sleep(5)
        upload_container.empty()  
        st.session_state.data_prep_ = True

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Insight Demo")
    st.markdown("""---""")
    st.session_state.raw_transcript_ = st.text_area(label="Enter Transcript Here", height=500)
    submit = st.button(label="Submit")
    if st.session_state.submit_ == False:
        st.session_state.submit_ = submit
    st.markdown("""---""")
    if st.session_state.submit_ and st.session_state.raw_transcript_ != "":
        prepare_workspace(st.session_state.raw_transcript_)
        st.title("Tools")
        st.session_state.curr_tool_ = st.selectbox(label="Select Tool", options=("Insights", "Search", "Question Answering"))
        if st.session_state.curr_tool_ == "Insights":
            insight_generate(st.session_state.raw_transcript_)
        if st.session_state.curr_tool_ == "Search":
            search()
        if st.session_state.curr_tool_ == "Question Answering":
            qna()
    elif st.session_state.submit_ and st.session_state.raw_transcript_ == "":
        st.error("Paste Transcript...")
