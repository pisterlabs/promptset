
import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

openai.api_key = os.environ.get("OPENAI_API_KEY")

# LLM，也就是我们使用哪个大语言模型，来回答我们提出的问题。在这里，我们还是使用 OpenAIChat，也就是 gpt-3.5-turbo 模型。
llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

# PromptTemplate 可以定义一个提示语模版，里面能够定义一些可以动态替换的变量。比如这个模版里，我们就定义了一个叫做 question 的变量，因为我们每次问的问题都会不一样
en_to_zh_prompt = PromptTemplate(
    template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
)

question_prompt = PromptTemplate(
    template = "{english_question}", input_variables=["english_question"]
)

en_to_cn_prompt = PromptTemplate(
    input_variables=["english_answer"],
    template="请把下面这一段翻译成中文： \n\n{english_answer}?",
)

question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")
# english = question_translate_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
# print(english)

qa_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="english_answer")
# english_answer = qa_chain.run(english_question=english)
# print(english_answer)

answer_translate_chain = LLMChain(llm=llm, prompt=en_to_cn_prompt)
# answer = answer_translate_chain.run(english_answer=english_answer)
# print(answer)


# 这个 LLMChain 会自动地链式搞定
chinese_qa_chain = SimpleSequentialChain(
    chains=[question_translate_chain, qa_chain, answer_translate_chain], input_key="question",
    verbose=True)
answer = chinese_qa_chain.run(question="请你作为一个IT专家，介绍一下外国人如何在加拿大找到一份程序开发的工作")
print(answer)