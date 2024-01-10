from typing import Optional

import openai

from ExpDerive.nlp.compiler.myTypes import Func
openai.api_key = "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"

from .db import DB

class BaseClassifier():
    def __init__(self):
        self.db = DB(api_key="5837c25c-f54b-4dfc-9e4a-f00b694c0f67", environment="us-west4-gcp")
        self.index = self.db.get_index("name")
        self.namespace = None
        self.threshold = 0.96

    # def load_model(self):
    #     file = open(self.model_path, 'rb')
    #     model = pickle.load(file)
    #     file.close()
    #     return model
    
    # def is_expr(self, arg):
    #     response = self.model.predict([arg])
    #     return response[0]
    
    def get_embedding(self, arg):
        response = openai.Embedding.create(
            input=[arg],
            model="text-embedding-ada-002",
        )
        return response['data'][0]['embedding']
    
    def classify(self, arg: str, force=False):
        embedding = self.get_embedding(arg)
        candidate = self.index.query(vector=embedding, top_k=1, namespace=self.namespace, include_metadata=True).matches
        if len(candidate) == 0:
            return None
        score = candidate[0].score
        print(arg, candidate)
        if score > self.threshold or force:
            return candidate[0]
        return None

class ArgClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.namespace = "variables"

    # def load_model(self):
    #     file = open(self.model_path, 'rb')
    #     model = pickle.load(file)
    #     file.close()
    #     return model
    
    # def is_expr(self, arg):
    #     response = self.model.predict([arg])
    #     return response[0]
    def classify(self, arg, force=False):
        label = super().classify(arg, force=force)
        return label.metadata['label'] if label is not None else None


class FuncClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.namespace = "function"

    def classify(self, arg: Func, force=True):
        if arg.type == "infix":
            self.namespace = "operator"
        else:
            self.namespace = "function"
        func = super().classify(arg.name, force=force)
        arg.name = func.metadata['label']
        arg.custom = func.metadata['type'] == "custom"
        return arg