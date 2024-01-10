from langchain import PromptTemplate

template = """Вопрос: {question}

Ответ: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "Сколько лет необходимо чтобы выучить английский язык на 7 баллов IELTS?"

from langchain import HuggingFaceHub, LLMChain

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}

)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about NFL 2010
print(llm_chain.run(question))
