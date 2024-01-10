import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from .BERTClassifier import BERTClassifier
from .KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer
from .BERTDataset import BERTDataset

import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

PATH = os.path.abspath(__file__)[:-20]

device = torch.device("cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
weight = os.path.join(PATH, "model/train/model_state_dict_231018.pt")
model.load_state_dict(torch.load(weight, map_location=device), strict=False)

# 파라미터 설정
max_len = 256
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

openai_key = os.environ.get("MY_API_KEY")

emotions = {
    "슬픔": "sad.xlsx",
    "공포": "scary.xlsx",
    "분노": "anger.xlsx",
    "놀람": "surprised.xlsx",
    "행복": "happy.xlsx"
}


def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


def predict(predict_sentence):
    global probability
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for logits in out:
            logits = logits.detach().cpu().numpy()
            logits = np.round(new_softmax(logits), 3).tolist()
            probability = []
            for logit in logits:
                probability.append(np.round(logit, 3))

            emotion = np.argmax(logits)
            probability.append(emotion)

    return probability


# Getting Embeddings
def get_embedding(content, openai_key):
    openai.api_key = openai_key
    # JSON 데이터 생성
    data = {
        "text": content
    }
    # JSON 데이터를 문자열로 변환
    json_data = json.dumps(data)

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=json_data
    )

    embedding = response['data'][0]['embedding']
    return embedding


# cosine_similarity 함수
def cosineSimilarity(report_text, music_file_lyrics):
    content1 = report_text
    # dataframe -> list 변환
    music_file_lyrics_list = music_file_lyrics.values.tolist()
    cosine_similarity_list = []

    text1_embs = get_embedding(content1, openai_key)

    for rank, lyric in music_file_lyrics_list:
        if lyric != None:
            text2_embs = get_embedding(lyric, openai_key)

            cosine_sim = cosine_similarity([text1_embs], [text2_embs])[0][0]
            cosine_similarity_list.append([rank, cosine_sim])

    # 유사도를 기준으로 내림차순
    cosine_similarity_list.sort(key=lambda x: -x[1])
    return cosine_similarity_list


def music_recommend(text_sim, emotion):
    recommend_rank = text_sim[:3]
    if emotion in emotions:
        file_path = f"dataset/{emotions[emotion]}"
        df = pd.read_excel(file_path)
    else:
        df = None

    music_information = []
    for i in range(3):
        x = df[df['순위'] == recommend_rank[i][0]]
        if not x.empty:
            music_info_dict = {
                "title": x['제목'].values[0],
                "singer": x['가수'].values[0],
                "imageUrl": x['앨범이미지'].values[0]
            }
            music_information.append(music_info_dict)
    return music_information

def recommend(emotion, content):
    if emotion in emotions:
        file_path = f"dataset/{emotions[emotion]}"
        music_file_lyrics = pd.read_excel(file_path, usecols=["순위", "가사"]).replace({np.nan: None})
    else:
        music_file_lyrics = None

    text_sim = cosineSimilarity(content, music_file_lyrics)
    music = music_recommend(text_sim, emotion)
    return music


def getEmotion(content):
    emotion = predict(content)[-1]
    if emotion == 0:
        return "슬픔"
    elif emotion == 1:
        return "공포"
    elif emotion == 2:
        return "분노"
    elif emotion == 3:
        return "놀람"
    elif emotion == 4:
        return "행복"


if __name__ == "__main__":
    end = 1
    text_sim = []
    while end == 1:
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0":
            break

        text_sim = []
        report_text = sentence
        emotion = getEmotion(sentence)
        if emotion in emotions:
            file_path = f"dataset/{emotions[emotion]}"
            music_file_lyrics = pd.read_excel(file_path, usecols=["순위", "가사"]).replace({np.nan: None})
        else:
            music_file_lyrics = None
        text_sim = cosineSimilarity(report_text, music_file_lyrics)
        music = music_recommend(text_sim, emotion)
