from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader
import prompts # this is a file containing all the prompt templates used for the chains
load_dotenv()

# This program helps you fill specific holes in your resume by doing the following:
# --> (LLMRequestsChain) scrapes 5 trending AI news headlines from google
# --> (LLMChain) for each one, write a question, to which the answer is contained in the headline
# --> (LLMChain) for each one, write answer choices, one of which is the answer to the question

#specify the LLM base you want to use
llm = OpenAI(temperature=0)

# search eventbrite for free upcoming workshops on that skill
events_template = prompts.events_template
events_prompt = PromptTemplate(input_variables=["requests_result"],template=events_template)
events_chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=events_prompt))
title = "AI in the news"
inputs = {
    "url": "https://www.eventbrite.com/d/online/free--events/" +title.replace(" ", "-")+ "/?page=1"   
}
events_result = events_chain(inputs)['output']
print(events_result)


# come up with a descriptive title of a hypothetical skill workshop that would bridge the skill gap.
title_template = prompts.title_template
title_prompt = PromptTemplate(input_variables=["deficiency"], template=title_template)
title_chain = LLMChain(llm=llm, prompt=title_prompt)
title_inputs = {
    "deficiency": events_result
}
title_result = title_chain(title_inputs)['text']
print(title_result)