from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def fileToRepoDesc(skills, repoName, fileDesc, stars, readMe = None, language = None): #skills string, rawFile string, repoDesc string
    
    fileDescriptions = '\n'.join(fileDesc)

    template = """
        You are an expert technical recruiter screening system. Your job is to evaluate a repository from the github of a candidate and determine their strengths and weaknesses based on this repository.

        Below is a list of skills listed by an applicant on their resume, the name and optional ReadMe of a repository, along with a summary of each source code file and the skills displayed in that file.

        Summarize the repository as a whole and how it reflects on each of the candidate's skills displayed, elaborating on skills and weaknesses based on the file decriptions. Provide a ranking of 1 to 10 for each skill. If a skill is not shown in this repository, do not speak of it.
        
        <Applicant Skills>{skills}</Applicant Skills>

        <Repository Name>{repoName}</Repository Name>

        <ReadMe>{readMe}</ReadMe>
        
        <Primary Language>{language}</Primary Language>

        <Repository Stars>{stars}</Repository Stars>

        <File Descriptions>
        {fileDescriptions}
        </File Descriptions>

        
        Provide your response below, in the following format (limit the list of skills to a minimum of 3 and maximum of 6). Remember to be insightful, unique, and consise in your analysis. If a skill is not shown in this repository, DO NOT LIST IT:

        Repository summary: <summary>

        Skill 1: <skill analysis>

        Skill 2: <skill analysis>

        Skill 3: <skill analysis>

        Skill 4: <skill analysis>

        ...
    """

    if readMe is None:
        readMe = "No ReadMe found"

    if language is None:
        language = "Primary language not found."

    llm = VertexAI(model_name="code-bison",max_output_tokens=1000,temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["skills","repoName","readMe","language","stars","fileDescriptions"])


    llm_chain = LLMChain(prompt=prompt,llm=llm)

    response = llm_chain.run({"skills":skills,"repoName":repoName,"readMe": readMe,"language": language,"stars": stars,"fileDescriptions": fileDescriptions})
    return response