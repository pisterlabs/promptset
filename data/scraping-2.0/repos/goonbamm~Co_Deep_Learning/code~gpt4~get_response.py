import os
import time
import json
import openai

import pandas as pd

from tqdm.auto import tqdm


# LOAD CONFIG
with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)
    
openai.api_key = CONFIG['OPENAI_API_KEY']
OPENAI_MODEL_NAME = CONFIG['OPENAI_MODEL_NAME']


def request_chatgpt_api(prompt):    
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
            ]
        )
    
    id = response['id']
    token_usage = response['usage']['total_tokens']
    content = response['choices'][0]['message']['content']

    return id, token_usage, content


def get_prompt(requirement, model_type):
    assert MODEL_TYPE in [
        'GPT4, FULL INSTRUCTION, K=2, MANUAL',
        'GPT4, FULL INSTRUCTION, K=2, RANDOM',
        'GPT4, FULL INSTRUCTION, K=0',
        'GPT4, MINIMAL INSTRUCTION, K=0',
        'GPT4, NO INSTRUCTION, K=2, MANUAL',
        'GPT4, NO INSTRUCTION, K=4, MANUAL',
        'GPT4, NO INSTRUCTION, K=6, MANUAL'
        ]
    
    if model_type == 'GPT4, FULL INSTRUCTION, K=2, MANUAL':
        prompt_list = [
            '요구사항을 토대로 그림 생성 영어 프롬프트로 바꿔줘.',
            '- 그림 생성 프롬프트는 명사 형태여야 해.\n',
            '예시)',
            '요구사항: 1800년대 스타일로 아틀라스를 그린 스케치를 하얀 배경에 몇 마리의 독수리가 주변에 날고 있는 그림을 그려봐.',
            'model prompt: Tethering vintage sketch of Atlas with a few eagles flying around him, on a white backdrop, in the style of 1800s\n',
            '요구사항: 1920년대 멋진 라틴 남자가 반짝이는 은색 수트를 입고 어두운 연기 가득한 바에서 음료를 들고 기대어 있는 그림을 그려봐. 초자연적이고 상세한 그래픽 노블 코믹북 일러스트레이션 같이, 컬러풀하게 그려줘. 마이크 미뇌라, 테리 도드슨, 토머 하누카가 만든 캐릭터 아트 스타일로 해줘.',
            'model prompt: Sand Dune 1920s handsome latino man in shiny silver suit, leaning with a drink in a dark smoky bar, supernatural, highly detailed realistic graphic novel comic book illustration, colorful, by mike mignola, terry dodson and tomer hanuka, character art\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, FULL INSTRUCTION, K=0':
        prompt_list = [
            '요구사항을 토대로 그림 생성 영어 프롬프트로 바꿔줘.',
            '- 그림 생성 프롬프트는 명사 형태여야 해.\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, FULL INSTRUCTION, K=2, RANDOM':
        prompt_list = [
            '요구사항을 토대로 그림 생성 영어 프롬프트로 바꿔줘.',
            '- 그림 생성 프롬프트는 명사 형태여야 해.\n',
            '예시)',
            '요구사항: 미래에서 온 슈퍼 강아지를 그려봐. 금색 피부에 근육이 많이 보이고, 인간처럼 근육이 튼튼한 몸을 가지고 있어. 배경은 흰색이고 미래적이고 초현실적인 느낌을 내주면 좋아. 그림은 전신을 다 보여줘야 해. 그리고 3D로 그려줘.',
            'model prompt: Naïve Art American Bully dog, gold color skin, heavily ripped muscles, with human muscles,hulk type muscled body, white background, futuristic, Ultrarealistic, 8K, Photographic details, full body with legs, Unreal engine, 3D,\n',
            '요구사항: 어둡고 무서운 분위기의 도시 안에 버려진 농구장 사진을 찍어봐. 빨간 빛으로 어두운 그림자가 드리워져 있고, 녹슨 바스켓이 쓸쓸하게 매달려있어. 그리고 농구장에는 한때 활기차게 경기가 벌어졌던 흔적이 보여야 해. 농구장은 낡아서 선과 그래픽이 흐리고 벗겨져 있어. 카메라는 셔터 스피드 1/200, 조리개 f/2.8로 설정하고, ISO는 800으로 설정해.  그림 전체적으로 유령 같고 영화 같은 느낌을 주고, 잊혀진 도시의 잊혀진 공간의 아름다움과 애수를 담아내야 해. 그림의 구도는 농구장에 주목할 수 있도록 신중하게 구성되어야 하며, 시간의 흐름과 인간의 노력의 일시적인 성격에 대한 생각을 이끌어내야 해.',
            'model prompt: Date A hauntingly beautiful photograph of an abandoned basketball court in the heart of a run-down, futuristic city, captured with a Sony A7R IV camera and a 35mm f/1.4 lens. The court is the sole focus of the image, with the surrounding cityscape in the background shrouded in deep shadows and ominous red lighting, casting an eerie glow over the scene. The court is worn and weathered, its once-vibrant lines and graphics now faded and peeling. The rusted hoops hang forlornly from the backboards, a testament to a bygone era when the court was a hub of activity and competition. Despite its derelict state, the court still exudes a sense of energy and possibility, a reminder of the dreams and aspirations that were once realized on its surface. The camera is set up with an aperture of f/2.8 to create a shallow depth of field, highlighting the intricate details of the court and blurring the background to create a sense of isolation and abandonment. The ISO is set to 800 to capture the dramatic red lighting that bathes the court, casting deep shadows and emphasizing the texture and detail of the surface. The overall effect is haunting and cinematic, capturing the beauty and melancholy of a forgotten space in a forgotten city. The composition is carefully crafted to draw the viewers eye to the court, emphasizing its significance and inviting reflection on the passage of time and the transience of human endeavor.\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, MINIMAL INSTRUCTION, K=0':
        prompt_list = [
            '요구사항을 토대로 그림 생성 영어 프롬프트로 바꿔줘.\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, NO INSTRUCTION, K=2, MANUAL':
        prompt_list = [
            '예시)',
            '요구사항: 1800년대 스타일로 아틀라스를 그린 스케치를 하얀 배경에 몇 마리의 독수리가 주변에 날고 있는 그림을 그려봐.',
            'model prompt: Tethering vintage sketch of Atlas with a few eagles flying around him, on a white backdrop, in the style of 1800s\n',
            '요구사항: 1920년대 멋진 라틴 남자가 반짝이는 은색 수트를 입고 어두운 연기 가득한 바에서 음료를 들고 기대어 있는 그림을 그려봐. 초자연적이고 상세한 그래픽 노블 코믹북 일러스트레이션 같이, 컬러풀하게 그려줘. 마이크 미뇌라, 테리 도드슨, 토머 하누카가 만든 캐릭터 아트 스타일로 해줘.',
            'model prompt: Sand Dune 1920s handsome latino man in shiny silver suit, leaning with a drink in a dark smoky bar, supernatural, highly detailed realistic graphic novel comic book illustration, colorful, by mike mignola, terry dodson and tomer hanuka, character art\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, NO INSTRUCTION, K=4, MANUAL':
        prompt_list = [
            '예시)',
            '요구사항: 1800년대 스타일로 아틀라스를 그린 스케치를 하얀 배경에 몇 마리의 독수리가 주변에 날고 있는 그림을 그려봐.',
            'model prompt: Tethering vintage sketch of Atlas with a few eagles flying around him, on a white backdrop, in the style of 1800s\n',
            '요구사항: 1920년대 멋진 라틴 남자가 반짝이는 은색 수트를 입고 어두운 연기 가득한 바에서 음료를 들고 기대어 있는 그림을 그려봐. 초자연적이고 상세한 그래픽 노블 코믹북 일러스트레이션 같이, 컬러풀하게 그려줘. 마이크 미뇌라, 테리 도드슨, 토머 하누카가 만든 캐릭터 아트 스타일로 해줘.',
            'model prompt: Sand Dune 1920s handsome latino man in shiny silver suit, leaning with a drink in a dark smoky bar, supernatural, highly detailed realistic graphic novel comic book illustration, colorful, by mike mignola, terry dodson and tomer hanuka, character art\n',
            '요구사항: 여름 그래픽, 로드 트립 어드벤처, 계획자 액세서리, 레트로 컬러, 수제 그림, 여행 일러스트레이션, 수채화 요소가 있는 그림을 그려봐.',
            'model prompt: Topology Summer Graphics, Road Trip Adventure, Planner Accessories, Retro Colors, Hand-Drawn Artwork, Travel Illustrations, Watercolor Elements, single design, 8k, \n',
            '요구사항: 클로드 모네 같은 그림을 그려봐. 아름다운 일몰이 있는 길가에 긴 브레이드를 한 작은 소녀가 큰 바위 위에 앉아있어. 그녀는 멀리 바라보며 생각에 잠기고 있어.',
            'model prompt: Nature Photography A little girl with long braids sits on a large boulder at the end of a linden-lined avenue, with a beautiful sunset in the background. She gazes off into the distance, lost in thought as she takes in the stunning view before her, in the style of Claude Monet, \n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)

    elif model_type == 'GPT4, NO INSTRUCTION, K=6, MANUAL':
        prompt_list = [
            '예시)',
            '요구사항: 1800년대 스타일로 아틀라스를 그린 스케치를 하얀 배경에 몇 마리의 독수리가 주변에 날고 있는 그림을 그려봐.',
            'model prompt: Tethering vintage sketch of Atlas with a few eagles flying around him, on a white backdrop, in the style of 1800s\n',
            '요구사항: 1920년대 멋진 라틴 남자가 반짝이는 은색 수트를 입고 어두운 연기 가득한 바에서 음료를 들고 기대어 있는 그림을 그려봐. 초자연적이고 상세한 그래픽 노블 코믹북 일러스트레이션 같이, 컬러풀하게 그려줘. 마이크 미뇌라, 테리 도드슨, 토머 하누카가 만든 캐릭터 아트 스타일로 해줘.',
            'model prompt: Sand Dune 1920s handsome latino man in shiny silver suit, leaning with a drink in a dark smoky bar, supernatural, highly detailed realistic graphic novel comic book illustration, colorful, by mike mignola, terry dodson and tomer hanuka, character art\n',
            '요구사항: 여름 그래픽, 로드 트립 어드벤처, 계획자 액세서리, 레트로 컬러, 수제 그림, 여행 일러스트레이션, 수채화 요소가 있는 그림을 그려봐.',
            'model prompt: Topology Summer Graphics, Road Trip Adventure, Planner Accessories, Retro Colors, Hand-Drawn Artwork, Travel Illustrations, Watercolor Elements, single design, 8k, \n',
            '요구사항: 클로드 모네 같은 그림을 그려봐. 아름다운 일몰이 있는 길가에 긴 브레이드를 한 작은 소녀가 큰 바위 위에 앉아있어. 그녀는 멀리 바라보며 생각에 잠기고 있어.',
            'model prompt: Nature Photography A little girl with long braids sits on a large boulder at the end of a linden-lined avenue, with a beautiful sunset in the background. She gazes off into the distance, lost in thought as she takes in the stunning view before her, in the style of Claude Monet, \n',
            '요구사항: 산 꼭대기 동굴에서 보는 4K 애니메이션 스타일의 그림을 그려봐. 그리고 스파이더맨 옆에는 바위 위에 앉아 동굴 입구를 바라보는 귀여운 고양이가 있어. 도쿄의 비오는 밤 거리와 밝은 네온 라이트, 그리고 비행하는 스팀펑크 배들이 나오는 장면이 있어.',
            'model prompt: 4K lofi anime style from the view of a mountain top cave with spiderman next to hir is a cute cat sitting on the rocks looking out of the a circle cave entrance with a night scene of rainy tokyo streets with bright neon lights and flying steampunk ships, \n',
            '요구사항: 큰 동물원 입구를 두꺼운 페인트로 그려봐. 입체적으로 그려줘.',
            'model prompt: Impasto a grand zoo entrance, isometric,\n',
            f'요구사항: {requirement}',
            'model prompt: '
        ]
        
        prompt = '\n'.join(prompt_list)


    return prompt


def main(model_type):    
    # LOAD DATASET
    test_dataset = pd.read_csv('../evaluation/final_test_128.csv', index_col=False)

    # FOR SAVING
    openai_dict = {
        'id': list(),
        'token_usage': list(),
        'content': list()
    }
    
    # READY FOR TQDM
    progress_bar = tqdm(total=len(test_dataset), desc=model_type)

    for i, row in test_dataset.iterrows():
        # GET PROMPT
        prompt = get_prompt(row['requirement'], model_type)
        
        # TRY TO REQUEST OPENAI API
        try:
            id, token_usage, content = request_chatgpt_api(prompt)
            
            openai_dict['id'].append(id)
            openai_dict['token_usage'].append(token_usage)
            openai_dict['content'].append(content)
        
        except Exception as e:
            print(f'#{i} DATA ERROR!')
            print(f'BECAUSE WE HAVE ERROR, WE SLEEP 5 MINUTES FROM NOW.\n')
            time.sleep(300)
            
        progress_bar.update(1)

    gpt4_requests = pd.DataFrame.from_dict(openai_dict)
    record = pd.read_csv('../evaluation/record.csv', index_col=False)
    record[model_type] = gpt4_requests['content']
    record.to_csv('../evaluation/record.csv', index=False, encoding='utf-8')    
    
if __name__ == '__main__':
    MODEL_TYPE = 'GPT4, NO INSTRUCTION, K=6, MANUAL'
    main(model_type=MODEL_TYPE)
