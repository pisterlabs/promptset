from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def loadModel():
    return LlamaCpp(
        model_path="../llama-2-7b-chat.Q5_K_M.gguf",
        temperature=0.0,
        top_p=1,
        max_tokens=8192,
        verbose=True,
        # 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
        n_ctx=4096,
    )


def generatePromptTemplate():
    template = """You're a Kubernetes expert. You understand the question and can generate a good kubernetes manifest.
    Question: {question}"""

    return PromptTemplate(template=template, input_variables=["question"])


model = loadModel()
prompt = generatePromptTemplate()
output_parser = StrOutputParser()

chain = {"question": RunnablePassthrough()} | prompt | model | output_parser


# ask to prompt
# question_1 = "generate kubernetes nginx pod and give me to yaml format"
question_2 = "generate kubernetes nginx pod and configmap that has nginx.conf. nginx pod use nginx.conf configmap"
print(chain.invoke(question_2))
