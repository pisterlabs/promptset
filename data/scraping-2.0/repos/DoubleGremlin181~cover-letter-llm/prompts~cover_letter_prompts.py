from langchain.prompts import PromptTemplate

prompt_template_classic = PromptTemplate.from_template(
    """Given the following resume and job listing information, generate a cover letter as part of the job application. The cover letter should not contain any contact information (to or from) and only contain salutations and a body of four to five information dense lines using business causal language. You should highlight any overlap of technology, responsibility or domain present between the job listing and my experience while mentioning why I would be a good fit for the given role. You should use optimistic and affirmative language and end the message with a call to action. Be concise.
------------        
Resume(Assume that the first few lines are personal details such as name and contact information):
{resume}
------------
Job Listing:
{job_listing}"""
)

prompt_template_modern = PromptTemplate.from_template(
    """Given the following resume and job listing information, generate a message answering the question "Tell us about yourself?" as part of the job application. You should begin the message simply with "Hi, I'm <my name>, <a short tagline created for me>" and follow it up with one short information dense paragraph using business causal language. You should highlight any overlap of technology, responsibility or domain present between the job listing and my experience while mentioning why I would be a good fit for the given role. You should use optimistic and affirmative language and end the message with a call to action. Be concise. 
------------        
Resume(Assume that the first few lines are personal details such as name and contact information):
{resume}
------------
Job Listing:
{job_listing}
------------"""
)
