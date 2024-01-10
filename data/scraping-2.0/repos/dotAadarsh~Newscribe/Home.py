import streamlit as st 
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from streamlit_pills import pills
from tweets import tweets

st.set_page_config(
    page_title="Newscribe",
    page_icon="🗞️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://twitter.com/DotAadarsh',
        'Report a bug': "https://github.com/dotAadarsh/Newscribe/discussions/1",
        'About': "Newscribe is an AI-powered app that transcribes news videos into written blog posts, simplifying content creation for bloggers and news writers."
    })


openai.api_key = st.secrets["OPEN_AI_API"]

st.title("Newscriber")
st.caption("Bringing news to life, one story at a time.")

with st.sidebar:
    st.image("./assets/news.png")
    st.success("Newscribe is an AI-powered app that transcribes news videos into written blog posts, simplifying content creation for bloggers and news writers.")
    st.info("For this demo, I have integrated the FirstPost news channels youtube content")
    st.warning("Feedbacks are welcome. Connect with me on [Twitter](https://twitter.com/DotAadarsh)")

@st.cache_data
def get_keywords_description(keywords):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="provide me a small description in markdown for each of the following " + keywords,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    description = response["choices"][0]["text"]
    return description


@st.cache_data
def get_keywords(transcript):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Extract important keywords mentioned in the following transcript: " + transcript,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    keywords = response["choices"][0]["text"]
    return keywords


@st.cache_data
def generateImage(imagePrompt):
    try:
        response = openai.Image.create(
        prompt= imagePrompt,
        n=1,
        size="256x256"
        )

        image_url = response['data'][0]['url']

        return image_url

    except:
        return ""
@st.cache_data
def generateBlog(transcript):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a blog with max of 300 words in markdown format for the following transcript: " + transcript,
    temperature=0.5,
    max_tokens=300,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    blog = response["choices"][0]["text"]
    return blog

@st.cache_data
def get_transcript(video_id):
    transcript = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        for segment in transcript_list:
            transcript += segment['text'] + " "
    except:
        transcript = "Transcript not available for this video."
    return transcript

@st.cache_data
def get_videos():
    yt_api_key = st.secrets["yt_api_key"]
    youtube = build("youtube", "v3", developerKey=yt_api_key)
    channel_id = "UCz8QaiQxApLq8sLNcszYyJw"
    
    request = youtube.search().list(
        part="id,snippet",
        channelId=channel_id,
        maxResults=8,
        order="date",
        type="video",
        videoDuration="short"
    )
    
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        video_title = item["snippet"]["title"]
        # video_url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"title": video_title, "id": video_id})

    return videos

def main():

    videos = get_videos()

    for video in videos:

        transcript = get_transcript(video["id"])

        if transcript ==  "Transcript not available for this video.":
            continue

        with st.expander(video["title"], expanded=False):
            tab1, tab2, tab3 = st.tabs(["Blog", "Video", "Social"])
            transcript = get_transcript(video["id"])
            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    imagePrompt = video["title"].split("|")[0].strip()
                    image_url = generateImage(imagePrompt)
                    if image_url != "":
                        st.image(image_url)
                    else:
                        st.write("Image cannot be generated!")

                with col2:
                    keywords = get_keywords(transcript)
                    keyword_list = []
                    if keywords.startswith("Keywords: "):

                        keyword_list = keywords[10:].split(", ")
                    else:
                        keyword_list = keywords.split(", ")

                    selected = pills("keywords", keyword_list)

                st.write(generateBlog(transcript))

            with tab2:
                url = "https://www.youtube.com/watch?v=" + video["id"]
                st.video(url)
                st.header("Bytes")
                st.write(get_keywords_description(keywords))

            with tab3:
                hashtag = keyword_list[1]
                tweets(hashtag)


if __name__ == "__main__":
    main()
