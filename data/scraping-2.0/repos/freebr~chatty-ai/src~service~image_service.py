import openai
from logging import getLogger, Logger

from definition.cls import Singleton

class ImageService(metaclass=Singleton):
    api_key = {}
    logger: Logger
    def __init__(self, **kwargs):
        self.logger = getLogger("IMAGESERVICE")
        self.api_key = kwargs['api_key']

    def __real_query(self, prompt, style):
        """
        调用 OpenAI API 接口生成图片并返回 URL
        """
        try:
            if len(prompt) == 0: return
            input = prompt
            if style: input += f', {style} style'
            res = openai.Image.create(prompt=input, n=1, size='1024x1024')
            url = res['data'][0]['url']
            return url
        except Exception as e:
            self.logger.error('非数学绘画失败：%s', str(e))
            return ''

    def invoke(self, args):
        """
        调用服务并返回信息
        """
        results = []
        prompts = args.get('prompt', '')
        styles = args.get('style', '')
        if type(prompts) == str: prompts = [prompts]
        if type(styles) == str: styles = [styles]
        for index, prompt in enumerate(prompts):
            style = styles[index]
            result = self.__real_query(prompt=prompt, style=style)
            results.append(result)
        return results