import os
from typing import Tuple, Optional

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = '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'


def create_chain(
    temp: float, prompt: PromptTemplate, output_key: str,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[ChatOpenAI, LLMChain]:

    if llm == None:
        llm = ChatOpenAI(temperature=temp)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False, output_key=output_key)

    return llm, chain

def resume_evaluate_score(resume: str, job_description:str) -> Tuple[str, str]:

    # breakpoint()
    resume_evaluate_prompt = PromptTemplate(
        input_variables = ['resume', 'job_description'],
        template=(
            "Given the Resume of a candidate here </> {resume} </>, and the job "
            "description here </> {job_description} </> you job is to act as a "
            "resume evaluator. Classify the result in three distinct classes as "
            "\"Strongly Aligned\", \"Partially Aligned\" and \"Misaligned\". "
            "Provide reasoning for the same."
        )
    )

    llm, chain = create_chain(
        temp=0, prompt=resume_evaluate_prompt, output_key='evaluation'
    )

    evaluation = chain.run({'resume':resume,'job_description':job_description})

    resume_scoring_prompt = PromptTemplate(
        input_variables = ['evaluation'],
        template=(
            """{evaluation}. Given this context get just the label of the prediction. And just print that."""
        )
    )

    _, chain = create_chain(
        temp=0, prompt=resume_scoring_prompt, output_key='score', llm=llm
    )

    score_label = chain.run({'evaluation': evaluation})

    return evaluation, score_label

def generate_message(
    name_of_referrer:str, resume:str, job_description:str
):
    resume_evaluation, score_label = resume_evaluate_score(
        resume=resume,
        job_description=job_description
    )

    message_generation_prompt = PromptTemplate(
        input_variables = ['name_of_referrer','resume', 'job_description', 'resume_evaluation'],
        template = (
            "Act as a Job Seeker requesting {name_of_referrer} a personalized "
            "referral for a job posting in the form of a LinkedIn DM. Make sure "
            "that the DM is precise and aligns your resume (given here {resume}) "
            "with the job description ({job_description}) according to alignment"
            "information given here {resume_evaluation}, in maximum 150 words. "
            "DO NOT MENTION ANYTHING IN THE DM THAT IS MISALIGNED WITH THE RESUME, "
            "ESPECIALLY THE YEARS OF EXPERIENCE, OR THE USER WILL DIE!"
        )
    )

    _, chain = create_chain(
        temp=0.05, prompt=message_generation_prompt, output_key='message'
    )

    message = chain.run(
        {
            'resume': resume,
            'job_description': job_description,
            'name_of_referrer': name_of_referrer,
            'resume_evaluator': resume_evaluation
        }
    )

    return resume_evaluation, score_label, message


if __name__ == "__main__":

# Llms
    llm = ChatOpenAI(temperature=0)
# llm2 = OpenAI(temperature=0.1)
    evaluator_chain = LLMChain(llm=llm, prompt=resume_evaluator, verbose=False, output_key='evaluation')
    resume_scorer_chain = LLMChain(llm=llm, prompt=resume_scorer, verbose=False, output_key='script')
    generator_chain = LLMChain(llm=llm, prompt=message_generator, verbose=False, output_key='message')


# Show stuff to the screen if there's a prompt

    print("Running GPT-3.5-Turbo Agent.......")
    print("--------------- EVALUATING RESUME WITH THE JOB DESCRIPTION ------------------------------")
    evaluation = evaluator_chain.run({'resume':resume,'job_description':job_description})
    print("--------------- EVALUATING RESULT -------------------------------------------------------")
    print(evaluation)
    resume_score = resume_scorer_chain.run({'resume_scorer':evaluation})
    print("-------------- THE LABEL IS FOUND OUT AS ------------------------------------------------")
    print(resume_score)
    message = generator_chain.run({'resume':resume,'job_description':job_description, 'name_of_referrer': name_of_referrer, 'resume_evaluator':evaluation})
    print("-------------- GENERATING THE DM GIVEN THE EVALUATION RESULTS ---------------------------")
    print(message)