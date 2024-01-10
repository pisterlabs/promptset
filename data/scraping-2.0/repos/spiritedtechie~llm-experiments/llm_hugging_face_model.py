from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
import os
from dotenv import load_dotenv

load_dotenv(".env")

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

oneQuestion = "What is the capital city of India?"

qa = [
    {"question": "What is the capital city of France?"},
    {"question": "What is the largest mammal on Earth?"},
    {"question": "Which gas is most abundant in Earth's atmosphere?"},
    {"question": "What color is a ripe banana?"},
]

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(oneQuestion))
print(llm_chain.generate(qa))
