from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain



def evaluator(title, description, llm):
    evaluationtemplate ="""You are an AI assistant tasked with assisting a hiring manager in enhancing job descriptions provided by the HR. The HR will provide you with a job title and description, and your goal is to score the job description based on the job title and provide recommendations for improvements. You will then give the HR the option to either continue with the original version or incorporate the suggested changes.

    To accomplish this, you will see the following information:

    Input:
    - Job Title: {title}
    - Job Description: {description}

    Your output should be in the form of recommendations and proposed changes to the job description. You can suggest improvements in language, emphasize important skills or qualifications, or provide additional details that would enhance the appeal of the job description.

    Remember to be respectful and tactful in your recommendations, while also demonstrating your superior technical knowledge to provide valuable enhancements.
    """
    eval_template = PromptTemplate(input_variables=["title", "description"], template=evaluationtemplate)
    evaluationChain = LLMChain(llm = llm, prompt=eval_template, output_key="job_description_evaluation")

    updatedDesctemplate ="""Job Description: {description}

    Proposed Enhancements and Recommendation:
    {job_description_evaluation}

    Updated Description:

    """
    updateddesc_template = PromptTemplate(input_variables=["description", "job_description_evaluation"], template=updatedDesctemplate)
    updatedChain = LLMChain(llm=llm, prompt=updateddesc_template, output_key="updated_job_description")

    enchancement_chain = SequentialChain(chains=[evaluationChain,updatedChain], input_variables=["title", "description"], output_variables=["job_description_evaluation","updated_job_description"], verbose=True)

    results = enchancement_chain({"title":title, "description": description })

    return results
