from typing import List
from core import config
from openai_api.openai_api import OpenAI_API
from openai_api.models import Model
from openai_api.text_generation.text_generation_request import TextGenerationRequestType
from openai_api.task_request.classification_request import ClassificationDomain, ClassificationRequest
import pandas as pd
from datasets import load_dataset


def args():
    model: Model = Model.DAVINCI
    
    request_type: TextGenerationRequestType = TextGenerationRequestType.COMPLETION
    
    labels: List[str] = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
    
    prompt_format = "다음 뉴스 기사 제목 \"{}\"이 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 중 어느 카테고리에 가장 잘 맞을까요?\n\n카테고리: "
    
    return model, request_type, labels, prompt_format


def init():
    api_key = config.OpenAIConfig.API_KEY
    if api_key is None:
        print('Please set OPENAI_API_KEY environment variable.')
        exit(-1)

    OpenAI_API(api_key=api_key)


def load_data():
    dataset = load_dataset("klue", "ynat")
    train_dataset = dataset["train"] # type: ignore
    test_dataset = dataset["validation"] # type: ignore
    return test_dataset


def write_predicts():
    model, request_type, labels, prompt_format = args()
    domain = ClassificationDomain(labels)
    api = ClassificationRequest(
        domain=domain,
        type=request_type,
        model=model,
        prompt_format=prompt_format,
    )
    data = load_data()
    filename = f"{model.value}_prompt-hash:{hash(prompt_format)}"
    
    result = []
    start_idx = 0
    
    # 기존 데이터로부터 복구
    try:
        previous_data = pd.read_csv(f"{filename}.csv")
        for i in range(len(previous_data)):
            json = {}
            for key in previous_data.keys():
                json[key] = previous_data[key][i]
            result.append(json)
        start_idx = len(result)
    except:
        start_idx = 0
    
    for i, (title, num_label) in enumerate(zip(data["title"], data["label"])): # type: ignore
        if i < start_idx:
            continue
        print(f"\r{i+1} / {len(data)}", end='')
        label = domain.label_idx_to_string(num_label)
        
        answer, prompt, raw_answer = api.request(title)
        
        json = {
            "title": title,
            "label": label,
            "answer": domain.label_idx_to_string(answer) if answer is not None else None,
            "prompt": prompt.replace("\n", "\\n"),
            "raw_answer": raw_answer,
        }
        result.append(json)
        pd.DataFrame(result).to_csv(f"{filename}.csv", index=False) # FIXME 오류로 종료되는 경우 임시로 해결

    pd.DataFrame(result).to_csv(f"{filename}.csv", index=False)


if __name__ == '__main__':
    init()
    
    import time
    while True:
        try:
            write_predicts()
            break
        except Exception as e:
            print(f"오류 발생 : {e}")
            print("10분 후 다시 시도합니다.")
            time.sleep(600)
            print()
