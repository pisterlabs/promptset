
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain


def final_eval(llm, sheet):
    finalevaltemplate = """"You are a member of the hiring committee of your company. Shortlisted Candidates have been tested with screening questions. you will provided with the Questions and the answers given by the candidate for each question. Your task is to evaluate their performance  for consideration in the next in-person round.
    The following is the Answer Script with questions and their answers by the candidate:
    {sheets}
    Your task is to evaluate the answers for each of the question and mark the each answer between 0-10. Then you should get the total marks obtained by the candidate and then you should output the names of the top 2 candidates who should be considered for next round of conversation. 

    Your Response should follow the following format:
    [{{"Name":"Name of the candidate", "Marks":"Total Marks goes here", "shortlisted": true}},
    {{"Name":Name of the candidate, "Marks":"Total Marks goes here", "shortlisted": true}},
    {{"Name":Name of the candidate, "Marks":"Total Marks goes here", "shortlisted": false}},]

    Do not output anything other than the JSON object."""

    finaleval_template = PromptTemplate(input_variables=["sheets"], template=finalevaltemplate)
    finalevalChain = LLMChain(llm = llm, prompt=finaleval_template, output_key="selected")

    finaleval_chain = SequentialChain(chains=[finalevalChain], input_variables=["sheets"], output_variables=["selected"], verbose=True)

    scores = finaleval_chain({"sheets": sheet})
    return scores