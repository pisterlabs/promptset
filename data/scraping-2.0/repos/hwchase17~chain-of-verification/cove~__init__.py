from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Import prompts
from .prompts import *

# Set up LLM to user
llm = ChatOpenAI(temperature=0)

# Chain to generate initial answer
baseline_response_prompt_template = PromptTemplate.from_template(BASELINE_PROMPT_WIKI)
baseline_response_chain = baseline_response_prompt_template | llm | StrOutputParser()

# Chain to generate a question template for verification answers
verification_question_template_prompt_template = PromptTemplate.from_template(VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI)
verification_question_template_chain = verification_question_template_prompt_template | llm | StrOutputParser()

# Chain to generate the verification questionts
verification_question_generation_prompt_template = PromptTemplate.from_template(VERIFICATION_QUESTION_PROMPT_WIKI)
verification_question_generation_chain = verification_question_generation_prompt_template | llm | StrOutputParser()

# Chain to execute the verification
execution_prompt_self_llm = PromptTemplate.from_template(EXECUTE_PLAN_PROMPT_SELF_LLM)
execution_prompt_llm_chain = execution_prompt_self_llm | llm | StrOutputParser()
verification_chain = RunnablePassthrough.assign(
    split_questions=lambda x: x['verification_questions'].split("\n"),
) | RunnablePassthrough.assign(
    answers = (lambda x: [{"verification_question": q} for q in x['split_questions']])| execution_prompt_llm_chain.map()
) | (lambda x: "\n".join(["Question: {} Answer: {}\n".format(question, answer) for question, answer in zip(x['split_questions'], x['answers'])]))# Create final refined response

# Chain to generate the final answer
final_answer_prompt_template = PromptTemplate.from_template(FINAL_REFINED_PROMPT)
final_answer_chain = final_answer_prompt_template | llm | StrOutputParser()


# Putting everything together, a final chain

chain = RunnablePassthrough.assign(
    baseline_response=baseline_response_chain
) | RunnablePassthrough.assign(
    verification_question_template=verification_question_template_chain
) | RunnablePassthrough.assign(
    verification_questions=verification_question_generation_chain
) | RunnablePassthrough.assign(
    verification_answers=verification_chain
) | RunnablePassthrough.assign(
    final_answer=final_answer_chain
)
