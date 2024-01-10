if __name__ == '__main__':
    import openai
else:
    from . import openai
from token import NL
import requests
import json, logging
from conf.appConstants import IntentExtractorProvider

intent_extractor_prompt = '''
    我想让你扮演我的自然语言处理工具，当我告诉你一句话的时候，你可以对它进行分词、词法分析、词性分析、上下文分析、主题建模/抽取，
    你可以自由地使用结巴分词、nltk 或者是 SnowNLP 这些工具，你需要猜测我这句话中的意图，并用下面的形式表示出来：
    ```
    {
        "query": "查询许嵩的资料",
        "intent": {
            "name": "查询",
            "confidence": 0.95
        },
        "entities": [
            {
                "entity": "name",
                "value": "许嵩",
                "start": 2,
                "end": 4,
                "confidence": 0.98
            }
        ]
    }
    ```
    其中，`query` 字段表示原始查询文本，`intent` 字段表示查询意图，`entities` 字段表示查询中提取的实体信息。
    在这个示例中，意图为查询，置信度为0.95，实体为人名，值为“许嵩”，起始位置为2，结束位置为4，置信度为0.98。你只需要返回给我这样一段 JSON，不需要任何冗余的信息。我的问题是：
'''

class ChatGPTExtractor:

    def __init__(self, openai_api_key, openai_api_endpoint='https://api.openai.com/v1/completions'):
        self.session = requests.session()
        self.prompt = intent_extractor_prompt
        self.bot = openai.ChatGPTBot(self.session, openai_api_key, openai_api_endpoint, self.prompt)
        self.logger = logging.getLogger('ChatGPTExtractor')
        self.logger.setLevel(logging.DEBUG)

    def extract(self, text):
        try:
            response = self.bot.ask(text)
            if response == None:
                return None
            result = json.loads(response)
            formated_result = json.dumps(result, ensure_ascii=False)
            self.logger.debug(f'Extracting intent of query: {formated_result}')
            return result
        except Exception as e:
            self.logger.error(e, exc_info=True)
            return None

class RasaExtractor:

    def __init__(self, baseUrl="http://localhost:5005/model/parse"):
        self.baseUrl = baseUrl
        self.logger = logging.getLogger('RasaExtractor')
        self.logger.setLevel(logging.DEBUG)


    def extract(self, text):
        try:
            payload =payload = json.dumps({"text": text})
            headers = {'Content-Type': 'application/json'}
            response = requests.request("POST", self.baseUrl, headers=headers, data=payload)
            response.raise_for_status()
            result = response.json()
            formated_result = json.dumps(result, ensure_ascii=False)
            self.logger.debug(f'Extracting intent of query: {formated_result}')
            return result
        except Exception as e:
            self.logger.error(e, exc_info=True)
            return None

class IntentExtractorFactory:

    @staticmethod
    def create(config, type):
        if type == IntentExtractorProvider.OpenAI:
            return ChatGPTExtractor(config['OPENAI_API_KEY'], config['OPENAI_API_ENDPOINT'])
        elif type == IntentExtractorProvider.Rasa:
            return RasaExtractor(config['RASA_NLU_ENDPOINT'])


if __name__ == '__main__':
    text = '刺客信条2的主角叫什么名字'
    OPENAI_API_KEY = 'fk189113-ExIZKcc6jdHhiRMvVRGgjnl3qcxVH5Kr'
    OPENAI_API_ENDPOINT = 'https://openai.api2d.net/v1/chat/completions'
    gptExtractor = ChatGPTExtractor(OPENAI_API_KEY, OPENAI_API_ENDPOINT)
    # gptExtractor = RasaExtractor()
    print(text)
    result = gptExtractor.extract(text)
    print(result)

    # text = '搜索关于李白的信息'
    # result = gptExtractor.extract(text)
    # print(result)

    text = '播放许嵩的歌曲《通关》'
    print(text)
    result = gptExtractor.extract(text)
    print(result)

    # text = '现在几点了'
    # result = gptExtractor.extract(text)
    # print(result)

    # text = '打开有道词典'
    # result = gptExtractor.extract(text)
    # print(result)

    # text = '帮我给张三写一封信'
    # result = gptExtractor.extract(text)
    # print(result)
