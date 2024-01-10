from fastapi import APIRouter,Depends,Response,HTTPException,status,BackgroundTasks
from .. import schemas,models,database,hashing
from .. import schemas,models,database
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import openai,os
from typing import List
import pandas as pd
import base64
import requests
router=APIRouter(
    prefix='/story',
    tags=['poem']
)


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
session_file = './storage/session.csv'
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

def generate_story(prompt: str):
    completion =openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Generate a delightful 4-5 paragraph children's story with an enchanting title about {prompt} that shares an important moral lesson. Our stories are meant to inspire and educate, so please choose a topic that's fun, safe, and suitable for kids. If the topic is irrelevant or inappropriate, share a special story with another topic instead."
            }
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    title = content.split('\n')[0]
    title = title.replace('Title: ', '')
    story = content[content.find('\n'):]
    story = story.lstrip()
    print(title, story)
    return title, story

def generate_image_prompt(story: str):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Give me a prompt for an image that would go with this story: {story}"}
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    if ':' in content:
        content = content[content.find(':')+1:]
    content = content.strip()
    return content

def generate_image(prompt: str):
    engine_id = "stable-diffusion-512-v2-1"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = os.getenv("STABILITYAI_API_KEY")
    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": f"{prompt}"
                }
            ],
            "cfg_scale": 6,  # Adjust scale as needed for detail
            "clip_guidance_preset": "FAST_BLUE",  # Use "FAST_BLUE" guidance
            "height": 512,  # Higher resolution for better quality
            "width": 512,
            "samples": 1,  # You can increase this for more variations
            "steps": 100,  # Increase steps for better convergence
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response for image generation: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):
        return image["base64"]
def generate_image2(prompt:str):

    api_key = "SG_7ce76e5b22d9ad5c"
    url = "https://api.segmind.com/v1/ssd-1b"

    # Request payload
    data = {
    "prompt": prompt,
    "negative_prompt": "bad anatomy,bad legs,bad hands, bad structure, bad face",
    "samples": 1,
    "scheduler": "UniPC",
    "num_inference_steps": 25,
    "guidance_scale": "9",
    "seed": "36446545871",
    "img_width": "1024",
    "img_height": "1024",
    "base64": True
    }

    response = requests.post(url, json=data, headers={'x-api-key': api_key})
    result = response.json()
    image_data = result.get('image')
    return image_data

def generate_image3(prompt:str):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": prompt,
    })
    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode()

    return image_base64

def generate_voice(story:str,file_name:str):
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    path = "/home/arseven/.local/lib/python3.10/site-packages/TTS/.models.json"

    model_manager = ModelManager(path)

    model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")

    voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

    syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=voc_path,
        vocoder_config=voc_config_path,
    )
   
    outputs = syn.tts(story)
    #make file name as id
    print("audio/"+file_name+".wav")
    syn.save_wav(outputs, "tale-genius/public/audio/"+file_name+".wav")
    return "True"
def get_followup_response(session_id: int, story_id: int, question: str,db):
    global session_df

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

    session_df = pd.concat([session_df, temp_df], ignore_index=True)
    session_df.to_csv(session_file, index=False)

    return content

@router.post("/generate/", response_model=dict)
def generate(story: schemas.prompt,background_tasks: BackgroundTasks,db:Session=Depends(database.get_db)):
    topic = story.prompt
    title, story_content = generate_story(topic)
    #read active_user file and get current active user
    file=open("active_user.txt","r")
    user=file.read()
    file.close()
    new_story=models.Story(title=title,body=story_content,username=user)
    db.add(new_story)
    db.commit()

    file_name=str(new_story.id)

    image_prompt = generate_image_prompt(story_content)
    image=generate_image3(image_prompt)
    with open("tale-genius/public/image/"+file_name+".png", "wb") as f:
        f.write(base64.b64decode(image))

    # generated_voice=generate_voice(story_content,file_name)
    background_tasks.add_task(generate_voice, story_content, file_name)
    return  {'sid': file_name,'title': title, 'story': story_content,
            'image': image,}
            # 'audio':generated_voice}
    # generated_voice=generate_voice(story_content,file_name)           

    story=db.query(models.Story).filter(models.Story.id==sid).first()
    story_content=story.body
    file_name=str(story.id)
    generated_voice=generate_voice(story_content,file_name)
    return {'audio':generated_voice}

@router.post("/question/", response_model=schemas.FollowUpResponse)
def get_followup(request: schemas.FollowUpRequest,db:Session=Depends(database.get_db)):
    session_id = request.session_id
    story_id = request.story_id
    question = request.question
    
    response = get_followup_response(session_id, story_id, question,db)
    # audio=generate_voice(response,'temp')
    return {'response': response,}


# @router.get('/{id}',status_code=200,response_model=schemas.ShowStory,   )
# def show(id,response: Response,db:Session=Depends(database.get_db),status_code=200):
#     blog=db.query(models.Blog).filter(models.Blog.id==id).first()
#     if not blog:
#         # response.status_code=status.HTTP_404_NOT_FOUND
#         # return {'detail':f'Blog with id {id} not found'}
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'Blog with id {id} not found')
#     else:
#         return blog

# @router.delete('/{id}',status_code=status.HTTP_204_NO_CONTENT, )
# def destroy(id,db:Session=Depends(database.get_db)):
#     blog=db.query(models.Story).filter(models.Blog.id==id) 
#     if not blog.first():
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'Blog with id {id} not found')
#     else:
#         blog.delete(synchronize_session=False)
#         db.commit()
#         return 'done'    

# @router.put('/{id}',status_code=status.HTTP_202_ACCEPTED,  )
# def update(id, request:schemas.Story,db:Session=Depends(database.get_db)):
#     blog=db.query(models.Blog).filter(models.Blog.id==id) 
#     if not blog.first():
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'Blog with id {id} not found')
#     else:
#         blog.update(request.dict ())
#         db.commit()
#         return 'updated'



@router.get('/title/{id}/', response_model=schemas.GetTitleResponse, status_code=status.HTTP_200_OK)
def show_title(id: int, db: Session = Depends(database.get_db)):
    blog = db.query(models.Story).filter(models.Story.id == id).first()
    if not blog:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Blog with id {id} not found')
    else:
        return {'title': blog.title}
@router.post('/body/{id}/', response_model=schemas.GetStoryBodyResponse, status_code=status.HTTP_200_OK)
def get_story_body(db: Session = Depends(database.get_db)):
    story = db.query(models.Story).filter(models.Story.id == d).first()
    if not story:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Story with ID {id} not found')
    return {'body': story.body}
