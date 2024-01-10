from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader
import prompts
load_dotenv()

# This program helps you fill specific holes in your resume by doing the following:
# --> scrapes a job description from linkedin, summarizes the most important core requirements
# --> loads your resume as a docx file
# --> compares your resume to the job description and highlights your top defficency with respect to the job description
# --> recommends a hypothetical skill workshop that would bridge the skill gap
# --> returns free upcoming workshops addressing that skill on eventbrite

#specify a job description url
jd_url = "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=3603193583"

#specify the LLM base you want to use
llm = OpenAI(temperature=0)

# load your resume file from the root directory
loader = Docx2txtLoader("resume.docx")
resume_doc = loader.load()
resume_text = resume_doc[0].page_content

# scrape linkedin for the job description, summarize it
jd_template = prompts.jd_template
jd_prompt = PromptTemplate(input_variables=["requests_result"], template=jd_template)
jd_chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=jd_prompt))
jd_inputs = {
    "url": jd_url
}
jd_response = jd_chain(jd_inputs)['output']
print(jd_response)


# compare the resume to the job description and highlight the top defficency
eval_template = prompts.eval_template
eval_prompt = PromptTemplate(input_variables=["jd_summary", "resume"], template=eval_template)
eval_chain = LLMChain(llm=llm, prompt=eval_prompt)
eval_inputs = {
    "jd_summary": jd_response,
    "resume": resume_text
}
eval_result = eval_chain(eval_inputs)['text']
print(eval_result)


# come up with a descriptive title of a hypothetical skill workshop that would bridge the skill gap.
title_template = prompts.title_template
title_prompt = PromptTemplate(input_variables=["deficiency"], template=title_template)
title_chain = LLMChain(llm=llm, prompt=title_prompt)
title_inputs = {
    "deficiency": eval_result
}
title_result = title_chain(title_inputs)['text']
print(title_result)


# search eventbrite for free upcoming workshops on that skill
events_template = prompts.events_template
events_prompt = PromptTemplate(input_variables=["requests_result"],template=events_template)
events_chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=events_prompt))
title = title_result
inputs = {
    "url": "https://www.eventbrite.com/d/online/free--events/" +title.replace(" ", "-")+ "/?page=1"   
}
events_result = events_chain(inputs)['output']
print(events_result)


# ~--------------------
# ~ things to try:
# ~ 1) [Beginner] Alter the prompts and record your observations. Ex- What happens if we don't use one-shot prompting? What happens if we increase temperature?
# ~ 2) [Beginner] Implement a different document loader for the resume. Ex- .pdf or .html instead of .docx
# ~ 3) [Intermediate] try implementing the pipeline as a single chain with SequentialChain https://python.langchain.com/docs/modules/chains/foundational/sequential_chains
# ~ 4) [Intermediate] try a different data pipeline. Instead of a resume, linkedin, and online workshops--> try a research paper, google trends, and google scholar
# ~ 5) [Advanced] Try a different LLM base model. Instead of OpenAI, try Llama-2, GPT-J, etc. using huggingface https://python.langchain.com/docs/integrations/llms/huggingface_hub
# ~ 6) [Advanced] Try to parse tabular data (.csv, sql, .xlsx) in a chain instead of text, and use the math chain to perform calculations using https://python.langchain.com/docs/modules/chains/additional/llm_math