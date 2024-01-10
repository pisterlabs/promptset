import openai
from diffusers import StableDiffusionPipeline
import torch
import re

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

def text_to_image_prompt_generator(song_title, artist): 
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=0.1,
        temperature=0.2,
        messages=[
            {"role": "system", "content":"You are an AI assistant designed to generate prompts for text-to-image models. When a user provides a song title and artist, you should summarize the song's lyrics in a single English sentence, and indicate its genre and mood."}, 
            {"role":"user", "content": f'{song_title} - {artist}'}
        ] 
    ) 
 
    return response.choices[0].message.content

# 주어진 텍스트에서 한글, 영문, 숫자가 아닌 문자를 밑줄(_)로 대체하는 함수
def replace_non_alphanumeric(text):
    pattern='[^가-힣a-zA-Z0-9]'
    result=re.sub(pattern, '_', text)
    return result

# 노래 제목과 아티스트를 입력받아 이미지를 생성하는 함수
def generate_dreamlike_image(song_title, artist):
    # 이미지 생성 모델의 ID 설정하기
    model_id="dreamlike-art/dreamlike-diffusion-1.0"
    
    # 모델 불러오기
    pipe=StableDiffusionPipeline.from_pretrained(model_id)
    # 모델을 GPU로 옮기기
    pipe=pipe.to("cuda") # 윈도우
    # pipe=pipe.to("mps") # 맥
    
    # 노래 제목과 아티스트를 사용해 노래 정보 생성하기
    about=text_to_image_prompt_generator(song_title, artist)
    
    # 이미지 생성에 사용할 프롬프트 작성하기
    prompt=f"""
    dreamlikeart,
    {about},
    dramatic lighting, illustration by greg rutkowski, yoji shinkawa, 4k, digital art, concept art, trending on artstation
    """
    
    # 이미지 생성에 사용할 부정적 프롬프트 작성하기
    negative_prompt="""
    text, deformed, cripple, ugly, additional arms, additional legs, additional head, two heads
    """
    
    # 파일명을 생성하고 특수 문자를 밑줄(_)로 대체하기
    file_name=replace_non_alphanumeric(f'{song_title} {artist}')
    
    # 이미지 생성하기
    image=pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    
    # 이미지를 저장할 경로 설정하기
    file_path=f"./dreamlike_diffusion/{file_name}.jpg"
    
    # 이미지 저장하기
    image.save(file_path)
    
    return file_path

if __name__ == '__main__':
    r=generate_dreamlike_image('Beat It', 'Michael Jackson')
    print(r)