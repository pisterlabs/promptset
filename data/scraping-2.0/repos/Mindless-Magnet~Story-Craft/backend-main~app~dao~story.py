import pandas as pd
import openai,os,re,base64,requests,site
from dotenv import load_dotenv

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

from .. import models

def followup_question_storage():
    session_file = './storage/session.csv'
    #create storage if not exists
    if not os.path.exists('./storage'):
        os.makedirs('./storage')

    if not os.path.exists(session_file):
        df = pd.DataFrame({
            "id": [],
            "sess_id": [],
            "story_id": [],
            "role": [],
            "content": []
        })
        df.to_csv(session_file, index=False)   

    session_df = pd.read_csv(session_file)
    return session_df

def generate_story(prompt: str,prespective:str="third"):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if prespective=="third":    
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a delightful 4 paragraph children's story with an enchanting title about {prompt} that shares an important moral lesson. Our stories are meant to inspire and educate, so please choose a topic that's fun, safe, and suitable for kids. If the topic is irrelevant or inappropriate, share a special story with another topic instead. Also create an image prompt for the story that mentions gender, physical features of characters and environment. Return the answer as a JSON object {{Title:, Prompt:, Story:,}}",
                }
            ]
        )
    else:
        print("first")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content":  f"Generate a delightful 4 paragraph children's story with an enchanting title about {prompt} that shares an important moral lesson. The story should be told entirely from a first-person perspective. Please use 'I,' 'Me,' and 'Myself' throughout the story, as if I am the hero of the tale. Our stories are meant to inspire and educate, so please choose a topic that's fun, safe, and suitable for kids. If the topic is irrelevant or inappropriate, share a special story with another topic instead. Also, create an image prompt for the story that mentions gender, physical features of characters, and the environment. Seperate story my double line.Return the answer as a JSON object {{Title:, Prompt:, Story:,}}."
 ,
                }
            ]
        )
            
    content = completion.choices[0].message.content

    json_string = content
    title_match = re.search(r'"Title": "([^"]+)"', json_string)
    prompt_match = re.search(r'"Prompt": "([^"]+)"', json_string)
    story_match = re.search(r'"Story": "([^"]+)"', json_string)
    print(json_string)

    if title_match and prompt_match and story_match:
        title = title_match.group(1)
        prompt = prompt_match.group(1)
        story = story_match.group(1)
        return title, prompt, story
    else:
        raise Exception("Could not find title, prompt, or story in response")
    
def generate_image_stability(prompt:str):
    load_dotenv()
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": HUGGINGFACE_API_KEY}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": prompt,
    })

    image_base64 = base64.b64encode(image_bytes).decode()

    return image_base64

def generate_image_segmind(prompt:str):
    load_dotenv()
    segmind= os.getenv("SEGMIND")
    url = "https://api.segmind.com/v1/ssd-1b"

    # Request payload
    data = {
    "prompt": prompt,
    "negative_prompt": "",
    "samples": 1,
    "scheduler": "UniPC",
    "num_inference_steps": 25,
    "guidance_scale": "9",
    "seed": "36446545871",
    "img_width": "1024",
    "img_height": "1024",
    "base64": True
    }

    response = requests.post(url, json=data, headers={'x-api-key': segmind})
    print(response)
    result = response.json()
    image_data=result['image']
    return image_data

def get_followup_response(session_id: int, story_id: int, question: str,db,session_df):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


    story = db.query(models.Story).filter(models.Story.id==story_id).first()
    system_msg = f"You are an assistant that answers the questions to the children's "\
                 "story given below. You should answer the questions descriptively in a "\
                 "way that a child can understand them. If the question asked is unrelated "\
                 "to the story, do not answer the question and instead reply by asking the "\
                 "user to ask questions related to the story."\
                 "\n\n"\
                 f"Story: {story.body}"

    temp_df = pd.DataFrame({
        "id": [len(session_df)+1],
        "sess_id": [session_id],
        "story_id": [story_id],
        "role": ["user"],
        "content": [question]
    })

    session_df = pd.concat([session_df, temp_df], ignore_index=True)

    messages = session_df[session_df['sess_id']
                          == session_id][["id", "role", "content"]]
    messages = messages.sort_values(by=['id'])
    messages = messages[['role', 'content']]
    messages = messages.to_dict('records')

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            *messages
        ]
    )

    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')

    temp_df = pd.DataFrame({
        "id": [len(session_df)+1],
        "sess_id": [session_id],
        "story_id": [story_id],
        "role": ["assistant"],
        "content": [content]
    })

    session_file = './storage/session.csv'
    session_df = pd.concat([session_df, temp_df], ignore_index=True)
    session_df.to_csv(session_file, index=False)

    return content,session_df

def generate_voice(story:str,file_name:str):

    location = site.getsitepackages()[0]
    path = location+"/TTS/.models.json"

    model_manager = ModelManager(path)

    model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/fast_pitch")

    voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

    syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=voc_path,
        vocoder_config=voc_config_path,
    )
   
    outputs = syn.tts(story)
    print("audio/"+file_name+".wav")
    syn.save_wav(outputs, "Story_Craft/public/assets/audio/"+file_name+".wav")
    return "True"

