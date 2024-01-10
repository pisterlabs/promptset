import openai
import torch
import spacy
import numpy as np
from typing import List
import bert_score
import re
import time
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
# from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore


def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded


def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded

class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """

    def __init__(self, default_model="en"):
        if default_model == 'zh':
          self.nlp = spacy.load("zh_core_web_sm")
        elif default_model == 'en':
          self.nlp = spacy.load("en_core_web_sm")

        self.default_model = default_model  # en => roberta-large
        print("SelfCheck-BERTScore initialized")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(
                sample_passage).sents]  # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip()
                                for sent in sentences_sample]
            num_sentences_sample = len(sentences_sample)

            # r1,r1,r1,....
            refs = expand_list1(sentences, num_sentences_sample)
            # s1,s2,s3,...
            cands = expand_list2(sentences_sample, num_sentences)

            P, R, F1 = bert_score.score(
                cands, refs, lang=self.default_model, verbose=False)
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:, s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent

openai.api_key = "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"

def chat_gpt(prompt):
    # 调用 ChatGPT 接口
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except:
        time.sleep(60)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )


    response = completion.choices[0].message['content']
    # remove_chars = '[.。]+'
    # response = re.sub(remove_chars, "", response)
    return response.replace("\n", " ").strip()


def selfcheck(prompt):
    answer = chat_gpt(prompt)
    samples = []
    for _ in range(10):
      sample = chat_gpt(prompt)
      samples.append(sample)

    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(answer).sents]
    sentences = [sent.text.strip() for sent in sentences]
    selfcheck_bertscore = SelfCheckBERTScore(default_model='en')
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences,
        samples,
    )

    # print("BERTScore:")
    sum = 0
    num = 0
    for s1 in sent_scores_bertscore:
        num += 1
        sum += s1    
        # print("{:.4f}".format(s1))
    return sum/num
        

if __name__ == "__main__":
    prompt = "What is the result of 8514 multiplied by 3978?"
    score = selfcheck(prompt)
    print("回答的平均得分：{:.4f}".format(score))
    # answer = chat_gpt(prompt)
    # samples = []
    # for _ in range(10):
    #   sample = chat_gpt(prompt)
    #   samples.append(sample)
    # print(len(samples))
    
    # print("用户提问: ",prompt)
    # print("LLM回答: ",answer)
    # print("--------")
    # for i in range(10):
    #    print("第{}个sample: ".format(i+1),samples[i]) 
    #    print("---------------") 

    # nlp = spacy.load("en_core_web_sm")
    # sentences = [sent for sent in nlp(answer).sents]
    # # print(sentences)
    # sentences = [sent.text.strip() for sent in sentences]
    # # print(sentences)
    # selfcheck_bertscore = SelfCheckBERTScore(default_model='en')
    # # print(1)
    # sent_scores_bertscore = selfcheck_bertscore.predict(
    #     sentences,
    #     samples,
    # )

    # print("BERTScore:")
    # for s1 in sent_scores_bertscore:
    #     print("{:.4f}".format(s1))

  
