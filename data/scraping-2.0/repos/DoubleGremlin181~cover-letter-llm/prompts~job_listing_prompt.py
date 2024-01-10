from langchain.prompts import PromptTemplate


prompt_template = PromptTemplate.from_template(
    "Given the text from a webpage of a job listing, extract the sections which contain information about the company and job inclduing but not limited to the overview, responsibilities and qualifications. Exclude any benfits information.:\n{raw_text}"
)
