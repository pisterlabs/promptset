from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector, MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings

from typing import Dict, List
import numpy as np

# 自定义选择器
def CustomExampleSelectorDemo():
    # 自定义选择器类
    class CustomExampleSelector(BaseExampleSelector):
        
        def __init__(self, examples: List[Dict[str, str]]):
            self.examples = examples
        
        def add_example(self, example: Dict[str, str]) -> None:
            """Add new example to store for a key."""
            self.examples.append(example)

        def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
            """Select which examples to use based on the inputs."""
            return np.random.choice(self.examples, size=2, replace=False)


    examples = [
        {"foo": "1"},
        {"foo": "2"},
        {"foo": "3"}
    ]

    example_selector = CustomExampleSelector(examples)
    example_selector.select_examples({"foo": "foo"}) # -> array([{'foo': '2'}, {'foo': '3'}], dtype=object)

    example_selector.add_example({"foo": "4"})
    example_selector.examples # -> [{'foo': '1'}, {'foo': '2'}, {'foo': '3'}, {'foo': '4'}]

    example_selector.select_examples({"foo": "foo"}) # -> array([{'foo': '1'}, {'foo': '4'}], dtype=object)


# LengthBasedExampleSelector Demo
# 此示例选择器根据长度选择要使用的示例。当您担心构建的提示会超过上下文窗口的长度时，这非常有用。对于较长的输入，它将选择较少的示例来包含，而对于较短的输入，它将选择更多的示例。
def LengthBasedExampleSelectorDemo():
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    
    example_selector = LengthBasedExampleSelector(examples=examples, example_prompt=example_prompt, max_length=25)
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:", 
        input_variables=["adjective"]
    )
    
    # 测试发现，此时会给出5个例子，因为每个例子的长度都小于25
    # print(dynamic_prompt.format(adjective="big"))
    
    # 测试发现，此时会给出1个例子
    long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
    print(dynamic_prompt.format(adjective=long_string))


# MaxMarginalRelevanceExampleSelector Demo 按最大边际相关性 (MMR) 选择示例
# 此示例选择器根据「最大边际相关性」选择要使用的示例。这是一种选择示例的方法，它尽可能地与已经选择的示例不同。这对于确保您的提示包含多种不同的示例非常有用。
# 例如，如果您正在构建一个提示，该提示将为您提供有关某个主题的信息，那么您可能希望确保提示包含多种不同的信息，而不是重复相同的信息。
# 该示例选择器使用了一个简单的启发式算法，该算法在每次选择示例时都会选择与已选择示例的平均相似度最低的示例。
# 请注意，这不是一个完美的算法，但它可以很好地工作，特别是在您的示例集中有许多不同的示例时。
def MaxMarginalRelevanceExampleSelectorDemo():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "angry", "output": "calm"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]
    
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(), # 这是用于生成嵌入的嵌入类，这些嵌入用于测量语义相似性。
        FAISS, # 这是用于存储嵌入并对其进行相似性搜索的VectorStore类。
        k=2,
    )
    mmr_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )
    
    print(mmr_prompt.format(adjective="worried"))
    
    # 让我们将其与仅通过相似性获得的内容进行比较，方法是使用SemanticSimilarityExampleSelector而不是MaxMarginalRelevanceExampleSelector。
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=2,
    )
    similar_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )
    print(similar_prompt.format(adjective="worried"))


# NGramOverlapExampleSelector Demo 按 n-gram 重叠选择示例
# 此示例选择器根据 n-gram 重叠选择要使用的示例。
# 例如，如果您正在构建一个提示，该提示将为您提供有关某个主题的信息，那么您可能希望确保提示包含多种不同的信息，而不是重复相同的信息。
# 该示例选择器使用了一个简单的启发式算法，该算法在每次选择示例时都会选择与已选择示例的平均相似度最低的示例。
# 请注意，这不是一个完美的算法，但它可以很好地工作，特别是在您的示例集中有许多不同的示例时。
def NGramOverlapExampleSelectorDemo():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    
    examples = [
        {"input": "See Spot run.", "output": "Ver correr a Spot."},
        {"input": "My dog barks.", "output": "Mi perro ladra."},
        {"input": "Spot can run.", "output": "Spot puede correr."},
    ]
    
    example_selector = NGramOverlapExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        threshold=-1.0,
        # 对于负阈值：
        #   选择器按ngram重叠分数对示例进行排序，并排除所有示例。
        # 对于大于1.0的阈值：
        #   选择器排除所有示例，并返回一个空列表。
        # 对于阈值等于0.0：
        #   选择器按ngram重叠分数对示例进行排序，并排除与输入没有ngram重叠的示例。
    )
    
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the Spanish translation of every input",
        suffix="Input: {sentence}\nOutput:",
        input_variables=["sentence"],
    )
    
    print(dynamic_prompt.format(sentence="Spot can run fast."))


# SemanticSimilarityExampleSelector Demo 按语义相似性选择示例
# 此示例选择器根据语义相似性选择要使用的示例。
def SemanticSimilarityExampleSelectorDemo():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=1
    )
    
    similar_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:", 
        input_variables=["adjective"],
    )
    
    print(similar_prompt.format(adjective="worried"))
    print(similar_prompt.format(adjective="fat")) # 貌似没生效
    
    similar_prompt.example_selector.add_example({"input": "enthusiastic", "output": "apathetic"})
    print(similar_prompt.format(adjective="joyful"))


if __name__ == "__main__":
    # CustomExampleSelectorDemo()
    # LengthBasedExampleSelectorDemo()
    # MaxMarginalRelevanceExampleSelectorDemo()
    # NGramOverlapExampleSelectorDemo()
    SemanticSimilarityExampleSelectorDemo()