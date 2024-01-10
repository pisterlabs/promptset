from papago import get_translate
from openAi import summarize_text_openai
import os
import nltk

# model's maximum context length is 4097 tokens, so we need to split the text into chunks of 4097 tokens
# 토큰 길이 확인 후 openai 요청
def summary_openai(text):
    # token_word = nltk.word_tokenize(text)
    try:
        # print('openAI text------> ', text)
        # if len(token_word) > 4097:
        #     all_sum_openai = ''
        # for i in range(0, len(token_word), 4097):
        #     sum_openai = summarize_text_openai(text[i:i+4097])   # <class 'openai.openai_object.OpenAIObject'>
        #     all_sum_openai += sum_openai["choices"][0]["text"]
        # else:    
        #     sum_openai = summarize_text_openai(text)
        #     all_sum_openai = sum_openai["choices"][0]["text"]
        sum_openai = summarize_text_openai(text)
        all_sum_openai = sum_openai["choices"][0]["text"]
        print('all_sum_openai: >>>>>>>>>>>>', all_sum_openai)
        return all_sum_openai
    except:
        print('요약 실패')
        return None

def translate_openai_summary(text, name, page_num):
    # openai 요약을 한국어로 번역
    global trans_text
    try:
        trans_text = get_translate(text)
        if trans_text is None:
            trans_text = '번역 실패'
        print('trans_text: >>>>>>>>>>>>', trans_text)
        f=open('./result/{}/2.{}_요약본.txt'.format(name, name),'a',encoding='utf-8')
        #줄바꿈
        f.write('\n'+ page_num + '>>-----------------------------------------'+text+'\n'+'-----------------------------------------'+'\n'+trans_text+'\n'+'-----------------------------------------'+'\n')
        f.close()
        return trans_text
    except:
        print('번역 실패')
        return None       