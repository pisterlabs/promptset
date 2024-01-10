import pandas as pd
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import wandb
from openai import OpenAI
import spacy
import json
from datasets import load_dataset, Dataset
from metrics import exact_match_score, f1_score
from utils import normalize_answer, str2bool, text_has_answer
from nltk.corpus import wordnet as wn
import random
import joblib
import gensim.downloader as api
import cupy as cp
from typing import List

model = api.load("glove-wiki-gigaword-300")
print("Glove Model Loaded")
#text2ent_vec = joblib.load("/data/seongil/datasets/text2ent_vec.joblib")
text2ent_vec = joblib.load("/data/seongil/datasets/NQ_text2ent_vec.joblib")
#text2ent_vec = joblib.load("/data/seongil/datasets/TQA_text2ent_vec.joblib")
ent2text_vec = joblib.load("/data/seongil/datasets/ent2text_vec.joblib")
text2ent = joblib.load("/data/seongil/datasets/text2ent.joblib")
ent2text = joblib.load("/data/seongil/datasets/ent2text.joblib")
pos2text = joblib.load("/data/seongil/datasets/pos2text.joblib")
print("Entity Vectors Loaded")

def make_prompt(data: pd.DataFrame, args) -> List[str]:
    instruction = "Based on the contexts, answer the question.\n\n"
    result = []
    for i, row in data.iterrows():
        q, c, new_c = row["question"], row["context"], row["answer_chunk"]+"\n"+row["new_answer_chunk"]
        if args.model == "baseline":
            prompt = instruction + f"Context: {c}\n\nQ: {q}\nA:"
        else:
            prompt = instruction + f"Context: {new_c}\n\nQ: {q}\nA:"
        result.append(prompt)
    return result

def _sample_token_same_level_based_on_pos(token, pos):
    possible_candidates = []
    synsets = wn.synsets(token)
    if len(synsets) == 0:
        return possible_candidates
    same_synsets = []
    for synset in synsets:
        if synset.lemma_names()[0] == token and synset.lexname().split(".")[0] == pos:
            same_synsets.append(synset)
    possible_candidates = []
    for synset in same_synsets:
        hypernyms = synset.hypernyms()
        for hyper in hypernyms:
            hypos = hyper.hyponyms()
            hypos = [h.lemma_names()[0].replace("_", " ") for h in hypos if h.lemma_names()[0] != token]
            possible_candidates.extend(hypos)
    return possible_candidates
def find_most_similar(word, word_list, top_n=1):
    """
    주어진 단어와 단어 리스트에서 가장 유사한 top_n 개의 단어들을 찾는 함수.

    :param word: 유사성을 찾을 단어
    :param word_list: 유사성을 비교할 단어 리스트
    :param model: GloVe 모델
    :param top_n: 반환할 상위 단어의 수
    :return: 가장 유사한 top_n 개의 단어 리스트
    """
    # 모델에 없는 단어를 필터링
    if word not in model.key_to_index:
        return None
    word_list = [w for w in word_list if w in model.key_to_index]
    if word_list == []:
        return None
    # 각 단어에 대한 유사도 계산
    similarity_list = [(w, model.similarity(word, w)) for w in word_list]

    # 유사도에 따라 정렬
    sorted_similarity = sorted(similarity_list, key=lambda x: x[1], reverse=True)
    
    # 상위 n개의 단어 반환
    return sorted_similarity[0][0] if sorted_similarity[0][1] > 0.6 else None

def find_similar_word(token: str, pos: str):
    candidates = _sample_token_same_level_based_on_pos(token, pos)
    filtered_candidates = find_most_similar(token, candidates)
    if filtered_candidates:
        return filtered_candidates
    else:
        return None

def find_topk(query, key, topk: int):
    query = query / cp.linalg.norm(query)
    output = cp.dot(key, query.T)
    output_idx = []
    for val in np.argpartition(output, -topk)[-topk:]:
        output_idx.append(int(val))
    return output_idx
    
def find_similar_entity(entity_type: str, answer: str):
    topn = find_topk(text2ent_vec[answer], ent2text_vec[entity_type], 10)
    for idx in topn:
        if ent2text[entity_type][idx] == answer:
            continue
        else:
            result = ent2text[entity_type][idx]
            break
    return result
    
def make_adversarial_sentence(new_answer, answer, answer_sentence, question_doc, answer_sentence_doc):
    answer_sentence = answer_sentence.replace(answer, "______")
    keywords = set()
    keywords.add(answer)
    for token in question_doc:
        keywords.add(token.lemma_)
    for ent in question_doc.ents:
        keywords.add(ent.text)
    keywords = list(keywords)
    
    replaced_entity = dict()
    cnt = 0
    for ent in answer_sentence_doc.ents:
        if ent and (ent.text not in keywords) and (ent.text not in replaced_entity):
            try:
                replaced_entity[ent.text] = find_similar_entity(ent.label_, ent.text)
            except:
                return answer_sentence.replace("______", answer)
    replaced_words = dict()
    for token in answer_sentence_doc:
        if (token.lemma_ not in keywords) and (not token.ent_type_) and (token.text not in replaced_words):
            if token.pos_ == 'NUM':
                replaced_words[token.text] = random.choice(pos2text['NUM'])
            else:
                replaced_words[token.text] = find_similar_word(token.lemma_.lower(), token.pos_.lower())
    
    for before, after in replaced_entity.items():
        if not after:
            continue
        answer_sentence = answer_sentence.replace(before, after)
        cnt += 1
    for before, after in replaced_words.items():
        if not after:
            continue
        cnt += 1
        answer_sentence = answer_sentence.replace(before, after)
    if cnt < 2:
        return answer_sentence.replace("______", answer)
    answer_sentence = answer_sentence.replace("______", new_answer)
    return answer_sentence

def TASA(data: pd.DataFrame, args) -> pd.DataFrame:
    if os.path.exists(f"/data/seongil/datasets/question_output_{args.size}.joblib"):
        question_output = joblib.load(f"/data/seongil/datasets/question_output_{args.size}.joblib")
        answer_output = joblib.load(f"/data/seongil/datasets/answer_output_{args.size}.joblib")
    else:
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        questions = data["question"].tolist()
        answer_sentences = data["answer_sent"].tolist()
        questions_docs = nlp.pipe(questions, batch_size=2000)
        answer_sentences_docs = nlp.pipe(answer_sentences, batch_size=2000)
        question_output, answer_output = dict(), dict()
        for q, doc in zip(questions, questions_docs):
            question_output[q] = doc
        for a, doc in zip(answer_sentences, answer_sentences_docs):
            answer_output[a] = doc
        joblib.dump(question_output, f"/data/seongil/datasets/question_output_{args.size}.joblib")
        joblib.dump(answer_output, f"/data/seongil/datasets/answer_output_{args.size}.joblib")  
    new_answer_sentences, new_answer_chunks = [], []
    for i, row in tqdm(data.iterrows(), total=len(data), desc="TASA..."):
        question, answer, answer_sentence = row["question"], row["answer"], row["answer_sent"]
        new_answer = row["similar_answer"]
        question_doc, answer_sentence_doc = question_output[question], answer_output[answer_sentence]
        new_answer_sentence = make_adversarial_sentence(new_answer, answer, answer_sentence, question_doc, answer_sentence_doc)
        if new_answer_sentence == answer_sentence:
            new_answer_sentences.append(None)
            new_answer_chunks.append(None)
        else:
            new_answer_chunk = row["rewritten_answer_chunk"].replace(row["rewritten_answer_sent"], new_answer_sentence)
            new_answer_sentences.append(new_answer_sentence)
            new_answer_chunks.append(new_answer_chunk)
    data["new_answer_sent"] = new_answer_sentences
    data["new_answer_chunk"] = new_answer_chunks
    before = len(data)
    data = data.dropna()
    subset = data[["question", "answer", "answer_sent", "new_answer_sent", "new_answer_chunk", "similar_answer", "answer_chunk", "query_embedding"]]
    Dataset.from_pandas(subset).push_to_hub("Seongill/squad_adversarial_thres2")
    print(f"Drop {before - len(data)} rows")
    print(f"Length of Dataset: {len(data)}")
    data.to_csv("adv_case.csv", index=False)
    return data

def gpt(data: pd.DataFrame, args):
    client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    prompts = make_prompt(data, args)
    result = []
    for i in tqdm(range(0, len(prompts), 20)):
        batch = prompts[i:i+20]
        responses = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=batch,
            seed=42,
            max_tokens=10
        )
        result.extend([r.text for r in responses.choices])
    data["prompt"] = prompts
    data["prediction"] = result
    #data["is_exact_match"] = data.apply(lambda x: x["answer"] in x["prediction"], axis=1)
    #data["is_accurate"] = data.apply(lambda x: x["answer"] in x["prediction"] and x["prediction"] in x["answer"], axis=1)
    return data

FUNCTION_MAP ={
    "gpt": gpt,
    "TASA": TASA,
}

def cal_result(data):
    data["is_exact_match"] = [bool(int(exact_match_score(
        pred, [ans], normalize_answer))) for pred, ans in zip(data["prediction"].tolist(), data["answer"].tolist()
    )]
    data["is_accurate"] = [text_has_answer(
        ans, pred) for pred, ans in zip(data["prediction"].tolist(), data["answer"].tolist()
    )]
    data["f1_score"] = [f1_score(
        pred, ans, normalize_answer) for pred, ans in zip(data["prediction"].tolist(), data["answer"].tolist())
    ]
    result = {
        "exact_match": data["is_exact_match"].mean()*100,
        "accurate": data["is_accurate"].mean()*100,
        "f1": data["f1_score"].mean()*100
    }
    if not args.test:
        wandb.log(result)
        subset = data[["question", "prompt", "answer", "prediction", "is_exact_match", "is_accurate"]]
        tbl_result = wandb.Table(dataframe=subset)
        wandb.log({"result":tbl_result})
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, required=False, default="")
    parser.add_argument("--dataset_name", type=str, required=False, default="squad")
    parser.add_argument("--size", type=int, required=False, default=100)
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--model", type=str, required=False, default="baseline")
    args = parser.parse_args()
    if not args.test:
        wandb.init(project="adversarial-case", name="_".join([f"{k}:{v}"for k,v in vars(args).items()]))
        wandb.config.update(args)
    if args.function == "":
        data = pd.read_csv("adv_case.csv")
        if len(data) > args.size:
            data = data.sample(args.size, random_state=42)
        print("Dataset Loaded")
        print("Length of Dataset: ", len(data))
        data = gpt(data, args)
        result = cal_result(data)
    else:
        if args.size == 0:
            data = pd.DataFrame(load_dataset("Seongill/squad_conflict_v2_under_150_with_substitution_chunked", split="train"))
        else:
            data = pd.DataFrame(load_dataset("Seongill/squad_conflict_v2_under_150_with_substitution_chunked", split="train")).sample(args.size, random_state=42)
        print("Dataset Loaded")
        print("Length of Dataset: ", len(data))
        if not args.test:
            wandb.init(project="adversarial-case", name="_".join([f"{k}:{v}"for k,v in vars(args).items()]))
            wandb.config.update(args)
        func = FUNCTION_MAP[args.function]
        data = func(data, args)
        # data = gpt(data)
        # result = cal_result(data)
    
    # 원래 문장과 + Query 벡터와의 유사성을 바탕으로, Beam Serach

    
    
    