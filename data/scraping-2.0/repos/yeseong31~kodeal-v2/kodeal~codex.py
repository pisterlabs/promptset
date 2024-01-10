import json

import openai


def get_answer_and_keyword(question, language):
    """
    OpenAI Codex Question -> Answer & Keyword
    """
    # OpenAI Codex의 입력으로 넣어줄 문장 구성
    pre_question = language + '\n' + question + 'with code'
    keyword = "Extract keywords from this text: " + question

    # OpenAI Codex의 반환값 중 필요한 데이터만 추출
    answer = extract_answer_sentences(question_to_codex(pre_question))
    keywords = extract_answer_sentences(question_to_codex(keyword))

    return {'answer': answer, 'keywords': separate_keywords_in_commas(keywords)}


def question_to_codex(question):
    """
    OpenAI Codex API
    """
    return openai.Completion.create(
        engine='text-davinci-002',  # Davinci 모델의 최신 버전(03.20)
        prompt=question,
        temperature=0.1,
        max_tokens=2000,  # Codex가 답할 수 있는 최대 문장 바이트 수
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )


def extract_answer_sentences(response):
    """
    OpenAI Codex의 반환 결과 중 사용자의 질문에 대한 답변만 추출
    """
    # 반환된 response 중에서 질문에 대한 답변이 포함된 'choices' 부분만 get
    choices = json.dumps(*response['choices'])

    # 위의 과정에서 choices의 값은 str 타입이기 때문에 JSON 형태로 변환해야 함
    json_choices = json.loads(choices)

    # JSON 형태로 변환된 문자열 중 키가 'text'인 값을 return
    answer = json_choices['text']

    # 전처리 과정을 거친 결과 반환
    return remove_unnecessary_char(answer)


# answer 문장 앞뒤로 불필요한 문자 제거
def remove_unnecessary_char(sentence):
    """
    OpenAI Codex 결과 값에 대한 전처리 수행
    """

    def remove_first_colon(answer):
        """
        첫 글자가 콜론(:)이라면 제거
        """
        if answer[0] in [':', '.']:
            return answer[2:]
        return answer

    def remove_two_newline_char(answer):
        """
        결과로 전달되는 answer 문장에서 맨 앞의 개행 문자 전처리
        """
        return answer.strip()

    sentence = remove_first_colon(sentence)
    sentence = remove_two_newline_char(sentence)
    return sentence


def separate_keywords_in_commas(keywords):
    """
    문장을 콤마(,) 단위로 분할
    """
    result = []
    for keyword in keywords.split(','):
        result.append(keyword.strip())
    return result
