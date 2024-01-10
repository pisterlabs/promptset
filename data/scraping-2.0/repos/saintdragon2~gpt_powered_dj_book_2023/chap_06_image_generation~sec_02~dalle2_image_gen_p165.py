from base64 import b64decode
import openai
import json
from pathlib import Path

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

def generate_dalle_image(prompt, image_file_name):
    DATA_DIR=Path.cwd() / 'dalle2_results' # 1 폴더 추가하기
    DATA_DIR.mkdir(exist_ok=True)
    print(DATA_DIR)

    response=openai.Image.create(
        prompt=prompt,
        n=1, # 몇 개의 이미지를 생성할지 정하기
        size="512x512", # 해상도(256×256, 512×512, 1024×1024 등 선택 가능) 
        response_format='b64_json' # 2 Base64 형태로 받기
    )

    # 3 파일명 생성하기
    file_name=DATA_DIR / f"{image_file_name}.png"

    # 응답에서 Base64로 인코딩된 이미지 데이터 추출하기
    b64_data=response['data'][0]['b64_json']
    
    # Base64 이미지 데이터를 디코딩해 바이너리 형식으로 변환하기
    image_data=b64decode(b64_data)
    
    # 이미지를 저장할 파일 경로 지정하기
    image_file=DATA_DIR / f'{file_name}'
    
    # 디스크에 이미지 파일 저장하기
    with open(image_file, mode='wb') as png:
        png.write(image_data)

    return image_file

if __name__=="__main__":
    generate_dalle_image('A man is dancing in the night in the middle of Gangnam, Seoul', 'dancing_man')
    # generate_dalle_image('Michael Jacksonis dancing in the night in the middle of Gangnam, Seoul', 'dancing_man')