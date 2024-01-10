from http import HTTPStatus
import dashscope
import config as c
from langchain.schema import BaseOutputParser
from typing import List

class QianwenLLM(BaseOutputParser[List[str]]):
    def parse(self, prompt: str):
        dashscope.api_key = c.ali_qw_key
        response = dashscope.Generation.call(
            model='qwen-max',
            prompt=prompt,
            seed=7656,
            top_p=0.8,
            result_format='message',
            enable_search=False,
            max_tokens=1500,
            temperature=1.0,
            repetition_penalty=1.0
        )
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message.content
            if content:
                print(content)
                # for model in c.audio_model_list:
                #     qwen_text_audio(content, model=model, output_path=rf'.\ali_qianwen\audio_result\{model}.mp3')
                return content
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return ""
        
if __name__ == "__main__":
    QianwenLLM.parse(" ".join(['dog', 'cat', 'horse', 'elephant', 'giraffe']))