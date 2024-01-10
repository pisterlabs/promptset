from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from PIL import Image
import io
import boto3
import json
import base64
import streamlit as st


model_name = "anthropic.claude-v2"


interest_to_link = {
    "Tech": {
        "link": "https://www.wsj.com/tech",
        "anchor": "Kajal", # works
        },
    "Politics": {
        "link": "https://www.wsj.com/politics",
        "anchor": "Brian",
        },
    "Finance": {
        "link": "https://www.wsj.com/finance",
        "anchor": "Amy", 
        },
    "Arts&Culture": {
        "link": "https://www.wsj.com/arts-culture",
        "anchor": "Lea",
        },
    "Business": {
        "link": "https://www.wsj.com/business",
        "anchor": "Joey", #works
        },
    "Sports": {
        "link": "https://www.wsj.com/sports",
        "anchor": "Joey", #works 
        },
    "Lifestyle": {
        "link": "https://www.wsj.com/lifestyle",
        "anchor": "Joanna",
        },
    "Personal Finance": {
        "link": "https://www.wsj.com/personal-finance",
        "anchor": "Amy",
        }
    # TODO erweitern
}


class NewsAnchor():
    def __init__(self, field, last, next=None) -> None:
        self.field = field
        self.last = last # bool
        self.next = next # next anchor
        self.name = interest_to_link[field]["anchor"]
        

    def set_next(self,next):
        self.next = next

class User:
    def __init__(self, first_name, last_name, interests, age=21, gender="Male", fav_celebs=[], sm_user=False):
        self.first_name = first_name
        self.last_name = last_name
        self.interests = interests
        self.age = age
        self.gender = gender
        self.fav_celebs = fav_celebs
        self.sm_user = sm_user
        generation = ""
        if age < 13:
            generation = "Children"
        elif age < 24:
            generation = "Gen-Z"
        elif age < 35:
            generation = "Young Adults"
        elif age < 60: 
            generation = "Adults"
        else:
            generation = "Senior citizens"
        self.generation = generation

    def __str__(self):
        return f"{self.first_name} {self.last_name}"



def _get_bedrock_llm():
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    llm = Bedrock(model_id=model_name, client=bedrock_client, region_name="us-east-1")
    return llm

def get_headlines(interest):
    # Load HTML
    links = [interest_to_link.get(interest,{"link":"https://www.wsj.com/"})["link"]]
    print(links)
    loader = AsyncChromiumLoader(links)
    html = loader.load()

    # Transform
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html
                                                       # , tags_to_extract=["h3"]
                                                       , tags_to_extract=["span"]
                                                        )
    # Result
    site = docs_transformed[0].page_content[0:5000]
    with open("templates/get_headlines.txt") as f:
        prompt_headlines = f.read()
    prompt_template = PromptTemplate.from_template(
        prompt_headlines
    )
    prompt = prompt_template.format(num_headlines=3, interest=interest, site_text=site)
    llm = _get_bedrock_llm()
    res = llm(prompt)
    return res


def get_newsfeed(user):
    headlines = []
    anchors = []
    generation = user.generation

    gender = user.gender
    if user.gender == "Other":
        gender = ""

    with open("templates/personalize_news_anchor.txt") as f:
        news_update = f.read()
    prompt_template = PromptTemplate.from_template(
        news_update
        )
    
    news = []
    audios = []
    for i in user.interests:
        cur_headlines = get_headlines(interest=i)
        last = (i==user.interests[-1])
        print(last)
        a = NewsAnchor(i, last)
        if len(anchors) > 0:
            anchors[-1].set_next(a)
        anchors.append(a)
        prompt = prompt_template.format(headlines=cur_headlines, generation=generation, name=user.first_name, age=user.age, gender = gender)
        llm = _get_bedrock_llm()
        print(prompt)
        print("===========================")
        res = llm(prompt)
        audio = t2speech(res, a)
        news.append(res)
        audios.append(audio)



    return news,audios

    
def t2speech(text, speaker):
    text = text.replace("\n", ".\n")
    client = boto3.client("polly")
    response = client.synthesize_speech(
        Engine='neural',
        LanguageCode='en-US',
        OutputFormat='mp3',
        Text=text,
        VoiceId=speaker.name
    )
    file = open('speech.mp3', 'wb')
    audio = response['AudioStream'].read()
    file.write(audio)
    file.close()
    return audio



def get_list_as_json(list):
    llm = _get_bedrock_llm()

    res = llm(f"""
        Give me this list of headlines in json format:
        
        {list}
    """)
    return res


def get_article_summary(articles, links):
    loader = AsyncChromiumLoader(links)
    html = loader.load()
    print(html)
    return articles[0]

def get_subway_surf_vid(topic, audio_clip, template_vid_path):
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(template_vid_path)
    audio_background = mpe.AudioFileClip(audio_clip)
    if "kika" in template_vid_path:
        final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background.set_start(10)])
    else:
        final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(final_audio)
    # from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    # # ffmpeg_extract_subclip("full.mp4", start_seconds, end_seconds, targetname="cut.mp4")
    # ffmpeg_extract_subclip("full.mp4", 60, 300, targetname="cut.mp4")

    output_path = topic + 'vid_result.mp4'
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

def gen_img(news_topic, audio, user):
    prompt = "Generate an image according to this topic: \n\n" + news_topic
    negative_prompts = [
        "poorly rendered",
    ]
    style_preset = "photographic"  # (e.g. photographic, digital-art, cinematic, ...)
    clip_guidance_preset = "FAST_GREEN" # (e.g. FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST)
    sampler = "K_DPMPP_2S_ANCESTRAL" # (e.g. DDIM, DDPM, K_DPMPP_SDE, K_DPMPP_2M, K_DPMPP_2S_ANCESTRAL, K_DPM_2, K_DPM_2_ANCESTRAL, K_EULER, K_EULER_ANCESTRAL, K_HEUN, K_LMS)
    width = 768
    request = json.dumps({
        "text_prompts": (
            [{"text": prompt, "weight": 1.0}]
            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
        ),
        "cfg_scale": 5,
        "seed": 452345,
        "steps": 60,
        "style_preset": style_preset,
        "clip_guidance_preset": clip_guidance_preset,
        "sampler": sampler,
        "width": width,
    })
    modelId = "stability.stable-diffusion-xl"

    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = bedrock_client.invoke_model(body=request, modelId=modelId)
    response_body = json.loads(response.get("body").read())

    print(response_body["result"])
    base_64_img_str = response_body["artifacts"][0].get("base64")
    image_2 = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
    im_name = "image"+news_topic+".png"
    image_2.save(im_name)

    return im_name

if __name__ == "__main__":
    news = get_newsfeed("Tech", 8)
    print(news)


def get_streamlit(user):
    if user == None:
        return None
    cur_user = user
    if cur_user is not None:
        st.write("## Welcome to your news feed ", cur_user)
        st.write("### From our data, we have analyzed that you are interested in ", str(cur_user.interests))
        st.write("We have prepared a custom news feed for you!")

        with st.spinner("Fetching newsfeed🤖"):
            news,audio = get_newsfeed(cur_user)
            for i in range(len(news)):
                st.write(news[i])
                st.audio(audio[i])
                try:
                    if cur_user.generation == "Gen-Z":
                        file = open(cur_user.interests[i]+'.mp3', 'wb')
                        file.write(audio[i])
                        file.close()
                        news_topic = cur_user.interests[i]
                        vid_file = get_subway_surf_vid(news_topic,news_topic+".mp3", 'media/subway_surf.mp4')
                        video_file = open(vid_file, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    elif cur_user.generation == "Children":
                        file = open(cur_user.interests[i]+'.mp3', 'wb')
                        file.write(audio[i])
                        file.close()
                        news_topic = cur_user.interests[i]
                        vid_file = get_subway_surf_vid(news_topic,news_topic+".mp3", 'media/kikaninchen.mp4')
                        video_file = open(vid_file, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    else:
                        st.image(gen_img(cur_user.interests[i], audio[i], cur_user))
                except Exception:
                    pass