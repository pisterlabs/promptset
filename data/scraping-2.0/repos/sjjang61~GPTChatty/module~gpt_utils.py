import openai
import os
from module.cache_utils import CacheManager

# openai.api_key = "OPENAPI_KEY"
text_length = 60
OPENAI_MODEL = "gpt-3.5-turbo"

gpt_cache_manager = CacheManager('url')

def set_api_key( key ):
    """
    OpenAI API KEY 설정
    :param key:
    :return:
    """
    openai.api_key = key

def get_article_summary( url : str, article_text : str, request : str ) -> dict:
    cache_data = gpt_cache_manager.get_cache( url )
    if cache_data != None:
        return cache_data['article']

    article_summary = train_article_summary( url, article_text, request )
    gpt_cache_manager.set_cache( { 'url' : url, 'article' : article_summary } )
    return article_summary


def train_article_summary(url, article_text, request) -> str:
    """
    기사 텍스트를 학습
    :param article_text:
    :param request:
    :return:
    """

    # 기사 텍스트를 학습시키는 prompt
    pre_sentence = """
    Summarize the following article in three simple sentences in "+setting_language +".
    After printing your summary, print '### example conversation ###'.
    Provide an additional example of a conversation between two 20-somethings based on the article.
    Organize them into 10 consecutive questions and answers.
    """
    #keyword = "and Let us know which "+setting_level+" words in the article caught your eye. Please wrap each word in the form 'word: interpretation' and Wrap every other word "
    prompt = pre_sentence + "\n\n" + article_text
    # ChatGPT에 대화 학습 요청
    try:
        response = openai.ChatCompletion.create(
            model = OPENAI_MODEL,
            messages=[
                {"role": "system", "content": request},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # 학습된 모델 반환
        return response['choices'][0]['message']['content']

    except Exception as e:
        print( "[Exception] ", e )


def get_chatbot_response(article_text, question, request) -> str:

    # 질문을 ChatGPT에 전달하는 prompt
    prompt = question + "\n\n"

    # ChatGPT에 질문 전달
    try:
        response = openai.ChatCompletion.create(
            model= OPENAI_MODEL,
            messages=[
                {"role": "system", "content": request},
                {"role": "system", "content": article_text},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=235,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0

        )

        # ChatGPT의 답변 반환
        return response['choices'][0]['message']['content']
    except Exception as e:
        print( "[Exception] ", e )

def get_chatbot_response_comple(article_text, question, request):

    # 질문을 ChatGPT에 전달하는 prompt
    prompt = question + "\n\n" + request

    # ChatGPT에 질문 전달
    try:
        response = openai.ChatCompletion.create(
            model = OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
        )

        # ChatGPT의 답변 반환
        return response['choices'][0]['message']['content']
    except Exception as e:
        print( "[Exception] ", e )
