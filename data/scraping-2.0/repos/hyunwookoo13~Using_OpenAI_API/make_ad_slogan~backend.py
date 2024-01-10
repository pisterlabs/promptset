# fastapi api 서버
# openai api를 활용해서 만든 광고 문구 작성 함수를 호출
from fastapi import FastAPI
from pydantic import BaseModel
import openai

openai.api_key = "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"


class SloganGenerator:
    def __init__(self, engine='gpt-3.5-turbo'):
        self.engine = engine
        self.infer_type = self._get_infer_type_by_engine(engine) # or completion
    
    def _get_infer_type_by_engine(self, engine):
        if engine.startswith("text-"):
            return 'completion'
        elif engine.startswith("gpt-"):
            return 'chat'
        
        raise Exception(f"Unknown engine type: {engine}")

    def _infer_using_completion(self, prompt):
        response = openai.Completion.create(engine=self.engine,
                                            prompt=prompt,
                                            max_tokens=200,
                                            n=1)
        result = response.choices[0].text.strip()
        return result

    def _infer_using_chatgpt(self, prompt):
        system_instruction = "assistant는 마케팅 문구 작성 도우미로 동작한다. user의 내용을 참고하여 마케팅 문구를 작성해라"
        messages = [{"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model=self.engine, messages=messages)
        result = response['choices'][0]['message']['content']
        return result

    def generate(self, product_name, details, tone_and_manner):
        prompt = f"제품 이름: {product_name}\n주요 내용: {details}\n광고 문구의 스타일: {tone_and_manner} 위 내용을 참고하여 마케팅 문구를 만들어라."
        if self.infer_type == 'completion':
            result = self._infer_using_completion(prompt=prompt)
        elif self.infer_type == 'chat':
            result = self._infer_using_chatgpt(prompt=prompt)
        return result


app = FastAPI()

class Product(BaseModel):
    product_name: str
    details: str
    tone_and_manner: str

@app.post("/create_ad_slogan")
def create_ad_slogan(product: Product):
    slogan_generator = SloganGenerator("gpt-3.5-turbo")

    ad_slogan = slogan_generator.generate(product_name=product.product_name,
                                          details=product.details,
                                          tone_and_manner=product.tone_and_manner)
    return {"ad_slogan": ad_slogan}