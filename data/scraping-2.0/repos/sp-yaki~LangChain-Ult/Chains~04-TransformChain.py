from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain, TransformChain

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

llm = ChatOpenAI()

yelp_review = open('../some_data/yelp_review.txt').read()

print(yelp_review)

def transformer_fun(inputs: dict) -> dict:
    '''
    Notice how this always takes an inputs dictionary.
    Also outputs a dictionary. You can call the output and input keys whatever you want, 
    just make sure to reference it correct in the chain call.
    '''
    # GRAB INCOMING CHAIN TEXT
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output':lower_case_text}

transform_chain = TransformChain(input_variables=['text'],
                                 output_variables=['output'],
                                 transform=transformer_fun)

template = "Create a one sentence summary of this review:\n{review_text}"
prompt = ChatPromptTemplate.from_template(template)
summary_chain = LLMChain(llm=llm,
                     prompt=prompt,
                     output_key="review_summary")

sequential_chain = SimpleSequentialChain(chains=[transform_chain,summary_chain],
                                        verbose=True)

result = sequential_chain(yelp_review)
print(result)