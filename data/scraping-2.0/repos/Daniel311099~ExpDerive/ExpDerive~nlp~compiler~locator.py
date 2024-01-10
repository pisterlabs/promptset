from dataclasses import dataclass
import openai
from nltk import word_tokenize

from ExpDerive.nlp.compiler.utils import parseArgs, parseFunc

from .myTypes import Func

class VarModel():
    # method to insert the delimiters around the function
    def process(self, phrase: str) -> str:
        return phrase


class VarGPTModel(VarModel):
    def __init__(
            self, 
            api_key="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", 
            engine="davinci:ft-personal-2023-06-04-17-44-42"
        ) -> None:
        openai.api_key = api_key
        self.engine = engine

    def process(self, phrase: str) -> str:
        response = self.gpt_call(phrase)
        return response.choices[0].text

    def gpt_call(self, phrase: str):
        template = "Annotate the arguments from the following expression using @@ to mark the start and ## to mark the end: "
        response = openai.Completion.create(
            engine=self.engine,
            prompt=template+phrase+" ->",
            max_tokens=100,
            temperature=0.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return response


class ArgLocator():
    def __init__(self, model: VarModel):
        # self.arg = arg
        self.model = model

    # def locate(self, phrase: str, type: str, funcLoc: tuple[int, int]):
    #     # get embedding of phrase
    #     # tokenize phrase
    #     # concat tokens with embedding
    #     embedding = self.getEmbedding(phrase[funcLoc[0]:funcLoc[1]])
    #     tokenized = self.tokenize(phrase[funcLoc[0]:funcLoc[1]])
    #     concat = embedding + tokenized
    #     if type == 'infix':
    #         return []
    #     elif type == 'prefix':
    #         return

    def getArgs(self, phrase, func: Func):
        if func.type == 'infix':
            words = phrase.split(" ")
            return [
                " ".join(words[:func.loc[0]]),
                " ".join(words[func.loc[1]:])
            ]
        processed = self.model.process(phrase)
        print("processed", processed)
        args = parseArgs(processed)
        return args




class FuncModel():
    # method to insert the delimiters around the function
    def process(self, phrase: str) -> str:
        return phrase


class FuncGPTModel(FuncModel):
    def __init__(
            self, 
            api_key = "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", 
            engine="davinci:ft-personal-2023-06-04-15-08-21"
        ) -> None:
        openai.api_key = api_key
        self.engine = engine

    def process(self, phrase: str) -> str:
        response = self.gpt_call(phrase)
        processed = response['choices'][0]['text']
        return processed

    def gpt_call(self, phrase):
        template = "Annotate the root function from the following expression using @@ to mark the start and ## to mark the end: "
        return openai.Completion.create(
            engine=self.engine,
            prompt=template+phrase+" ->",
            max_tokens=100,
            temperature=0.2,
            top_p=1,
            stop=["\n"]
        )


class FuncLocator():
    def __init__(self, model: FuncModel):
        # self.arg = arg
        # self.inModel = None
        # self.preModel = None
        # self.sufModel = None
        self.model = model

    def getType(self, phrase):
        pass

    # def locate(self, phrase: str, type: str):
    #     # get embedding of phrase
    #     # tokenize phrase
    #     # concat tokens with embedding
    #     embedding = self.getEmbedding(phrase)
    #     tokenized = self.tokenize(phrase)
    #     concat = [embedding] + tokenized
    #     if type == 'infix':
    #         return [self.inModel.predictStart(concat), self.inModel.predictEnd(concat)]
    #     elif type == 'prefix':
    #         return self.preModel.predict(concat)
    #     elif type == 'suffix':
    #         return self.sufModel.predict(concat)

    def getFunc(self, phrase):
        processed = self.model.process(phrase)
        func = parseFunc(processed)
        print(processed, func)
        return func

    def getEmbedding(self, phrase):
        response = openai.Embedding.create(
            input=[phrase],
            model="text-similarity-ada-001",
        )
        return response['data'][0]['embedding']

    def tokenize(self, phrase):
        return word_tokenize(phrase)


