from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
import os

#from git_output_sample_data.json_nico import test_dict

#Input is summary of all the repos
# List of skills
# summarize all summaries

def RepoSumToEval(RepositorySummaries, skills): #skills string, rawFile string, repoDesc string
    
    RepositorySummaries = '\n\n'.join(RepositorySummaries)

    template = """
    You are an expert technical recruiter screening system. Your job is to use the summaries and evaluations of a list of repositories from a candidate's github to determine their strengths and weaknesses, and whether they are fit to be hired.

    Below is a list of skills the candidate has listed on their resume and a list of summaries of candidate repositories, along with evaluations of their technical skills. Use this information to make a final, consolidated judgement of this candidate's experience, strengths, and weaknesses. At the end of your description of the candidate, give your definitive reccomendation.

    <Skills>{skills}</Skills>

    <Repository Summaries>{repo_summaries}</Repository Summaries>

    Provide your response below in bullet-point format, with a maximum of three paragraphs. Be detailed but consise, provide constructive critisism where needed, do not be too harsh but do not be too generous, and provide unique, creative, and useful insight:
    
    """

    llm = VertexAI(model_name="text-bison",max_output_tokens=500,temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["repo_summaries","skills"])


    llm_chain = LLMChain(prompt=prompt,llm=llm)
    
    # Define StuffDocumentsChain
    #stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    #loader = TextLoader(RepositorySummaries)

    response = llm_chain.run({"repo_summaries":RepositorySummaries,"skills":skills})
    return response

if __name__ == "__main__":
    dummy_list = '[Relational Database Design, Data Modeling, SQL Programming, Data Integrity, Normalization, Natural Language Processing (NLP), Machine Learning, Artificial Intelligence (AI), Conversational AI, Chatbot Development, Command-Line Interface (CLI) Programming, User Experience Design, Software Development Methodologies, Task Automation, User Interaction]'
    with open(os.path.join(os.getcwd(),'chad_giga_repo.txt'),'r') as f:
        dummy_sum = f.read()
    print(RepoSumToEval(dummy_sum,dummy_list))
