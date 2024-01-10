import openai
from timeout_decorator import timeout, TimeoutError

# todo 本番運用ではソフトコーディングする
# OpenAI API Settings
openai.api_key = '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'
openai.organization = 'org-fgWaYosTeNaXwPemvIQvJdq4'
ai_model_engine = 'gpt-3.5-turbo'


# OpenAI ChatGPT
@timeout(30)
def ai_call(msg):
    """ OpenAI ChatCPT APIコール
    OpenAIのChatCPTをコールし文字列を返す。

    Args:
        msg (str): list of string
        AIとの会話を格納した配列。

    Returns:
        str: AIから返された文字列。
    """
    try:
        completion = openai.ChatCompletion.create(
            model=ai_model_engine,
            messages=msg
        )
        response = [0, completion.choices[0].message.content]
        return response
    except TimeoutError:
        response = [1, f'ChatGPT timeout precautionary step.']
        return response
    except Exception as e:
        response = [1, f'{e}']
        return response


if __name__ == '__main__':
    ai_call()
