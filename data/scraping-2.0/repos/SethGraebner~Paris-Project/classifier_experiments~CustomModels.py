'''
Custom model definitions to fit the sklearn schema so that they can be modular with the rest of the experiments
'''

from abc import ABC, abstractmethod
from type_utils import Label, Match
from typing import List, Tuple
import numpy as np
import openai
import dotenv
from numpy.typing import NDArray
import json

class AbstractModel(ABC):
    @abstractmethod
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> None:
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

class GPTModel():
    '''
    Experimental bespoke classifier using GPT {3, 4} to classify OCR snippets as 'good' or 'bad', zero shot.

    Note: I would assume that neither GPT model performs that well here, but it may be worth looking into providing few-shot examples or fine-tuning. However both of these options may be costly.
    '''
    def __init__(self, gpt4: bool = False):
        self.gpt4 = gpt4

        api_key = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

        if api_key is None:
            raise Exception("OPENAI_API_KEY not found in .env. This may be because you have not created a .env file.")
        
        openai.api_key = api_key

    def predict(self, X: List[Match]) -> List[Label]:
        return [self._get_label_from_gpt(x) for x in X]

    def score(self, X: List[Match], y_true: List[Label]) -> float:
        y_hat = self.predict(X)
        return float(np.mean(y_hat == y_true))
    
    def _get_label_from_gpt(self, x: Match) -> Label:
        msgs = []

        msgs.append({
            "role": "system",
            "content": f"""
            You are a talented French historian. The following text is an excerpt from a potentially faulty OCR scan of a 19th century French book containing mention of the monument {x['monument']}. You are tasked with determing whether the text is 'good' or 'bad', on the following criteria:
            - The text describes physical properties of the monument, such as its size, shape, or color
            - The text describes the location of the monument relative to other landmarks, or within the city of Paris

            However, these physical descriptions must be in regard to the specified monument {x['monument']} and NOT other misc physical descriptors.

            Again, note that due to OCR errors the text may contain spelling or formatting errors, but these should not hinder your analysis at all.

            You will return your prediction as a json object with the following format (without the supporting ellipses and newlines):
            ... {{
            ...     "label": "good" | "bad"
            ... }}

            Do not provide anything else except for the json object. If you do, your answer will be marked as incorrect.
            """
        })

        text = f'Here is the text: \n {x["snippet"]} \n Provide a JSON response and NOTHING ELSE, or you will be penalized'
        msgs.append({"role": "user", "content": text})

        # print(msgs)

        model = "gpt-4" if self.gpt4 else "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model,
            messages=msgs
        )

        model_response: str = response.choices[0].message.content # type: ignore
        print(model_response)
        # parse string into a dict
        model_json = json.loads(model_response)

        #Label[model_json['label']]

        return model_json['label']
        



class AlwaysBad(AbstractModel):
    '''
    Always predicts "bad," used for ablation testing
    '''
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> None:
        pass

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        return np.array([Label.bad] * X.shape[0])
    
    def get_params(self, deep = None) -> dict:
        return {}