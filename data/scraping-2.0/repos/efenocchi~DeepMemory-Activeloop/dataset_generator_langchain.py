from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

prompt_template = "What is a good name for a company that makes {product}?"

system_message = """
Generate a question related to the context and provide a relevance score on a scale of 0 to 1, where 0 is not relevant at all and 1 is highly relevant.  
The input is provided in the following format: 
Context: [The context that for the generated question] 
The output is in the following format: 
#Question#: [Text of the question] 
#Relevance#: [score number between 0 and 1] 
The context is: {context}

"""
system_message_output_answer = """
Generate an answer to the question provided in the input using the context provided. They will have the following format:
Context: ["Context1", "Context2", "Context3", "Context4"]
Question: [Text of the question]
The context is: {context}
The question is: {question}
"""


# TODO remove \n from the question
def get_chunk_qa_data_old(context):
    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(system_message))
    output = llm_chain(context)

    # CHECK THE RELEVANCE STRING IN THE OUTPUT
    check_relevance = None
    relevance_strings = ["#Relevance#: ", "Relevance#: ", "Relevance: ", "Relevance"]
    for rel_str in relevance_strings:
        if rel_str in output["text"]:
            check_relevance = rel_str
            break

    if check_relevance is None:
        raise ValueError("Relevance not found in the output")

    messages = output["text"].split(check_relevance)
    relevance = messages[1]

    # CHECK THE QUESTION STRING IN THE OUTPUT
    question = None
    question_strings = ["#Question#: ", "Question#: ", "Question: ", "Question"]
    for qst_str in question_strings:
        if qst_str in messages[0]:
            question = messages[0].split(qst_str)[1]
            break

    if question is None:
        raise ValueError("Question not found in the output")

    return question, relevance


def generate_answer(context, question):
    # llm = OpenAI(temperature=0)
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)

    formatted_message = system_message_output_answer.format(
        context=context, question=question
    )
    for chunk in llm.stream(formatted_message):
        print(chunk, end="", flush=True)
        yield chunk
    # llm_chain = LLMChain(
    #     llm=llm, prompt=PromptTemplate.from_template(system_message_output_answer)
    # )

    # output = llm_chain((context, question))
    # return output["text"]
