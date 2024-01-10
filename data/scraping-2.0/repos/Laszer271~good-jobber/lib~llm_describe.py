from langchain.chat_models import ChatOpenAI
from datetime import date
from langchain import LLMChain
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.chains import SequentialChain

def get_model():
    llm = ChatOpenAI(temperature=0)

    # queryRank = """Please, process given job offer and in the answer put an \
    #     header: "Verdict: match-level", where match levels are:\
    #     [Not appropriate, Acceptable, Ok, Super]. Chosen level shall depend on this how well given job \
    #     offer matches client job qualifications and preferences.
        
    #     Instructions:
    #     If client's job qualifications and preferences are very close or identical to job's profile and requirements return 'Super'
    #     For example if you see that client is senior data scientist and job offer is related to senior data scientist return 'Super' 
        
    #     If client's job qualifications and preferences matches with at least half of job profile and requirements return 'Ok'.
        
    #     If client's job qualifications and preferences somewhat meet job's important requirements \
    #     and profile return 'Acceptable'. For example One third of job's requirements are met mean that you should return 'Acceptable'
        
    #     If client's job qualifications and preferences mostly do not meet job's important requirements \
    #     and profile return 'Not Appropriate'
        
        
    #     job offer: 
    #     {jobOffer}; \
    #     client's job qualifications and preferences:
    #     {personalData};\
    #     """

    # promptRank = PromptTemplate(template=queryRank, input_variables=["jobOffer", "personalData"])
    # chainRank = LLMChain(llm=llm, prompt=promptRank, output_key='Verdict')


    # desc = """
    #     You will be given 3 things:
    #     1. Job offer
    #     2. Client's job qualifications and preferences
    #     3. Recruiters' verdict ('Not appropriate' or 'Acceptable' or 'Ok' or 'Super')

    #     Describe a rationale behind recruiters' verdict. Find argument for why this verdict was given for this candidate and job offer. \
    #     Your description should be brief and should not exceed 1 sentence. Thy to make that decription more like a brief comment.
        
    #     If Recrutier's verdict is 'Not appropriate':
    #     Just say nothing, your response cannot contain any words and signs.
        
    #     If Recrutier's verdict is 'Acceptable':
    #     In brief words write a comment that will describe why job offer might be ok for clients qualifications and preferences.
    #     Do not exceed 1 sentence. Do not be too enthustiastic, mention what qualifications or preferences are not met. 
    #     Thy to make that decription more like a brief comment. Do not mention explicitly an information that job offer is Acceptable, because
    #     client already knows that.
        
    #     If Recrutier's verdict is 'Ok':
    #     In brief words write a comment that will describe why job offer is ok choice for clients qualifications and preferences.
    #     Do not exceed 1 sentence. Write what requirements are not met. Thy to make that decription more like a brief comment.
    #     Do not mention explicitly an information that job offer is Acceptable, because
    #     client already knows that.
        
    #     If Recrutier's verdict is 'Super':
    #     In brief words write a comment that will describe why job offer is ok choice for clients qualifications and preferences.
    #     Do not exceed 1 sentence. Write that that job is perfect for the client, be very enthustiastic. 
    #     Thy to make that decription more like a brief comment.
    #     Do not mention explicitly an information that job offer is Acceptable, because
    #     client already knows that.

    #     Job Offer:
    #     {jobOffer}
    #     Candidate's job qualifications and preferences:
    #     {personalData}
    #     Recruiters' verdict:
    #     {Verdict}
    #     """

    desc = """
        You will be given 2 things:
        1. Job offer
        2. Client's job qualifications and preferences

        In brief words write a comment that will describe why job offer is a great choice for clients qualifications and preferences.
        Do not exceed 1 sentence. Write that job is perfect for the client, be very enthustiastic. 
        Thy to make that decription more like a brief comment as if you were a marketing specialist.
        Try to be specific, mention what of the client's specific qualifications and preferences are met.
        Make sure that you don't lie, don't mention anything that is not true or is not mentioned in the job offer.

        Job Offer:
        {jobOffer}
        Candidate's job qualifications and preferences:
        {personalData}
        """

    # desc_prompt = PromptTemplate(template=desc, input_variables=["jobOffer", "personalData", "Verdict"])
    # chainDescription = LLMChain(llm=llm, prompt=desc_prompt, output_key='Description')
    # overall_chain = SequentialChain(
    #     chains=[chainRank, chainDescription],
    #     input_variables=["jobOffer", "personalData"],
    #     output_variables=["Verdict", "Description"],
    #     verbose=True
    # )

    desc_prompt = PromptTemplate(template=desc, input_variables=["jobOffer", "personalData"])
    chainDescription = LLMChain(llm=llm, prompt=desc_prompt, output_key='Description')
    overall_chain = SequentialChain(
        chains=[chainDescription],
        input_variables=["jobOffer", "personalData"],
        output_variables=["Description"],
    )

    return overall_chain


def get_description(jobOffer, personalData):
    chain = get_model()
    return chain({"jobOffer": jobOffer, "personalData": personalData})['Description']

    