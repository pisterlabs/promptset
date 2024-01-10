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

def RepoSumToScores(RepositorySummaries): #skills string, rawFile string, repoDesc string
    
    RepositorySummaries = '\n\n'.join(RepositorySummaries)

    template = """
    You are an expert technical recruiter screening system. Your job is to use the summaries and evaluations of a list of repositories from a candidate's github to determine their proficiency with various skills.

    Below is a list of summaries of candidate repositories, along with evaluations of their technical skills. Use this information to deduce a numerical score (1 to 10) for each displayed skill.

    <Repository Summaries>{repo_summaries}</Repository Summaries>

    Provide your response below in json format, according to the following structure (only display the json table, with no additional formatting), display at most the 6 skills with the most evidence to support your conclusions:

    {{
        "<placeholder skill>": X.X,
        "<placeholder skill>": X.X,
        "s<placeholder skill>": X.X,
        "<placeholder skill>": X.X,
        "<placeholder skill>": X.X,
        "<placeholder skill>": X.X
    }}

    Response:
    
    {{
    """

    llm = VertexAI(model_name="text-bison",max_output_tokens=500,temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["repo_summaries"])


    llm_chain = LLMChain(prompt=prompt,llm=llm)
    
    # Define StuffDocumentsChain
    #stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    #loader = TextLoader(RepositorySummaries)

    response = llm_chain.run({"repo_summaries":RepositorySummaries})
    return response