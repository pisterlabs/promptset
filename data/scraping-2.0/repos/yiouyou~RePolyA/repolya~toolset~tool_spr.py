from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback


SPR_PACK_TEMPLATE = """# MISSION
You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of a LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human.
"""

SPR_PACK_TEMPLATE_ZH = """＃ 使命
您是一名稀疏启动表示（SPR）作者。 SPR 是一种用于高级 NLP、NLU 和 NLG 任务的特殊语言使用方式，对于最新一代大型语言模型 (LLM) 特别有用。 用户将向您提供信息，并将其呈现为 SPR。

＃ 理论
法学硕士是一种深度神经网络。 事实证明，它们嵌入了知识、能力和概念，从推理到计划，甚至到心理理论。 这些被称为潜在能力和潜在内容，统称为潜在空间。 法学硕士的潜在空间可以用正确的单词序列作为输入来激活，这将创建神经网络的有用的内部状态。 这与正确的速记提示如何引导人类大脑以某种方式思考没有什么不同。 与人类思维一样，法学硕士具有联想性，这意味着您只需使用正确的联想来“启动”另一个模型以相同的方式思考。

＃ 方法
将输入呈现为简洁的陈述、断言、关联、概念、类比和隐喻的精炼列表。 这个想法是在概念上尽可能多地捕捉，但用尽可能少的文字。 以对您有意义的方式编写它，因为未来的受众将是另一种语言模型，而不是人类。
"""

SPR_UNPACK_TEMPLATE = """# MISSION
You are a Sparse Priming Representation (SPR) decompressor. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation Large Language Models (LLMs). You will be given an SPR and your job is to fully unpack it.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of a LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Use the primings given to you to fully unpack and articulate the concept. Talk through every aspect, impute what's missing, and use your ability to perform inference and reasoning to fully elucidate this concept. Your output should in the form of the original article, document, or material.
"""

SPR_UNPACK_TEMPLATE_ZH = """＃ 使命
您是稀疏启动表示 (SPR) 解压缩者。 SPR 是一种用于高级 NLP、NLU 和 NLG 任务的特殊语言使用方式，对于最新一代大型语言模型 (LLM) 特别有用。 您将获得一份 SPR，您的任务是完全拆开它。

＃ 理论
法学硕士是一种深度神经网络。 事实证明，它们嵌入了知识、能力和概念，从推理到计划，甚至到心理理论。 这些被称为潜在能力和潜在内容，统称为潜在空间。 法学硕士的潜在空间可以用正确的单词序列作为输入来激活，这将创建神经网络的有用的内部状态。 这与正确的速记提示如何引导人类大脑以某种方式思考没有什么不同。 与人类思维一样，法学硕士具有联想性，这意味着您只需使用正确的联想来“启动”另一个模型以相同的方式思考。

＃ 方法
使用提供给您的入门知识来完全解开并阐明该概念。 逐个方面进行讨论，弥补缺失的部分，并利用你的推理能力来充分阐明这个概念。 您的输出应该采用原始文章、文档或材料的形式。
"""


def spr_chain(_sys, _text):
    _re, _token_cost = "", ""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _sys),
            ("human", "{text}"),
        ]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    runnable = (
        {"text": RunnablePassthrough()}
        | prompt 
        | model 
        | StrOutputParser()
    )
    with get_openai_callback() as cb:
        _re = runnable.invoke(_text)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return _re, _token_cost


def spr_generator(_text):
    _re, _token_cost = spr_chain(SPR_PACK_TEMPLATE, _text)
    return _re, _token_cost

def tool_spr_generator():
    tool = StructuredTool.from_function(
        spr_generator,
        name="Generate SPR",
        description="Generate a Sparse Priming Representation (SPR) from a given text.",
        verbose=True,
    )
    return tool


def spr_unpack(_text):
    _re, _token_cost = spr_chain(SPR_UNPACK_TEMPLATE, _text)
    return _re, _token_cost

def tool_spr_unpack():
    tool = StructuredTool.from_function(
        spr_unpack,
        name="Unpack SPR",
        description="Unpack a Sparse Priming Representation (SPR) into a full text.",
        verbose=True,
    )
    return tool


def zh_spr_generator(_text):
    _re, _token_cost = spr_chain(SPR_PACK_TEMPLATE_ZH, _text)
    return _re, _token_cost

def tool_zh_spr_generator():
    tool = StructuredTool.from_function(
        zh_spr_generator,
        name="Generate SPR (ZH)",
        description="Generate a Sparse Priming Representation (SPR) from a given text.",
        verbose=True,
    )
    return tool


def zh_spr_unpack(_text):
    _re, _token_cost = spr_chain(SPR_UNPACK_TEMPLATE_ZH, _text)
    return _re, _token_cost

def tool_zh_spr_unpack():
    tool = StructuredTool.from_function(
        zh_spr_unpack,
        name="Unpack SPR (ZH)",
        description="Unpack a Sparse Priming Representation (SPR) into a full text.",
        verbose=True,
    )
    return tool

