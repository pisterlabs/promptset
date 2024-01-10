from . import openai
import requests
import importlib

class NLPCorrector:

    def __init__(self):
        self.pycorrector = None
        try:
            self.pycorrector = importlib.import_module('pycorrector')
        except ImportError as e:
            print("pycorrector is required, run 'pip install pycorrector' first")

    def correct(self, text):
        corrected, details = self.pycorrector.correct(text)
        return corrected
    
class ChatGPTCorrector:
    
    def __init__(self, openai_api_key, openai_api_endpoint='https://api.openai.com/v1/completions'):
        self.session = requests.session()
        self.prompt = '''
            我想让你扮演一名写作专家，你可以使用任意自然语言处理工具对输入的文本进行纠错，
            你只需要返回纠正过以后的文本，不需要携带任何多余信息。我需要纠错的文本是：
        '''
        self.bot = openai.ChatGPTBot(self.session, openai_api_key, openai_api_endpoint, self.prompt)

    def correct(self, text):
        return self.bot.ask(text)


if __name__ == '__main__':
    text = '少先队员因该为老人让坐，买一个爱疯叉需要多少钱，蜘蛛侠士钢铁侠的儿子吗'

    nlpCorrector = NLPCorrector()
    result = nlpCorrector.correct(text)
    print(f'NLPCorrector:{text} -> {result}')

    OPENAI_API_KEY = ''
    OPENAI_API_ENDPOINT = ''
    gptCorrector = ChatGPTCorrector(OPENAI_API_KEY, OPENAI_API_ENDPOINT)
    result = gptCorrector.correct(text)
    print(f'ChatGPTCorrector:{text} -> {result}')
