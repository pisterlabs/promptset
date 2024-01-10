import re
import logging

from fastapi import APIRouter
from enum import Enum

from langchain import LLMChain

from dependencies import llm
from prompts import (
    chat_prompt,
    summary_prompt,
    validation_prompt,
    category_l1_prompt,
    category_l2_prompt,
    ticket_prompt,
    response_prompt
)
from prompt_example import (
    l1_category_names,
    l2_category_names,
    get_validation_examples,
    get_category_l1_examples,
    get_category_l2_all_examples,
    get_category_l2_example,
    get_ticket_examples,
    get_response_all_examples,
    get_response_example
    )

from translate import get_translation

logging.basicConfig(filename="clf_api.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

router = APIRouter()

class Language(Enum):
    en = "english"
    es = "spanish"
    ko = "korean"

CAT_L1_NAMES = l1_category_names()

VALID_DATA = get_validation_examples() 
CAT_L1_DATA = get_category_l1_examples()
TICKET_DATA = get_ticket_examples()

CAT_L2_DATA = get_category_l2_all_examples()
RESP_DATA = get_response_all_examples()

@router.get("/ticket_qualification")
async def categorize_and_respond(user_input: str, language: Language, memory: str = ""):
    # Combine the user's input and memory
    context = f"{memory}\n{user_input}"
    
    if memory != "":
        summarize_chain = LLMChain(llm=llm, prompt=summary_prompt)
        context = summarize_chain.run(context=context)

    chatter = LLMChain(llm=llm, prompt=chat_prompt)
    chat_resp = chatter.run(complaint=context)
    
    logger.info("Complaint summarization completed")
    
    # Validation
    validation_chain = LLMChain(llm=llm, prompt=validation_prompt)
    validation_result = validation_chain.run(examples=VALID_DATA, user_input=context)
    
    logger.info("Complaint validation completed")

    if validation_result == "incomplete":
        if language.value == "english":
            return {
                "status": 430,
                "response": chat_resp
                }
            
        to_translate = ["status", 430, "response", chat_resp]
        result = get_translation(to_translate, language.value)
        return result

    # Level 1 classification
    category_l1_chain = LLMChain(llm=llm, prompt=category_l1_prompt)
    category_l1 = category_l1_chain.run(examples=CAT_L1_DATA, user_input=context)
    category_l1 = re.sub("[^a-zA-Z]*", "", category_l1.lower())
    print(category_l1)
    
    logger.info("Complaint L1 classification completed")

    # Level 2 classification
    
    if category_l1 not in CAT_L1_NAMES:
        if language.value == "english":
            return {
                "status": 431,
                'response': "undefined level 1 category",
                "message": chat_resp
                }
        
        to_translate = ["status", 431, "response", "undefined level 1 category", "message", chat_resp]
        
        result = get_translation(to_translate, language.value)
        return result
    
    
    category_l2_eg = get_category_l2_example(CAT_L2_DATA, category_l1)
    category_l2_chain = LLMChain(llm=llm, prompt=category_l2_prompt)
    l2_categories = l2_category_names(category_l1)
    category_l2 = category_l2_chain.run(categories=l2_categories, examples=category_l2_eg, user_input=context)
    
    logger.info("Complaint L2 classification completed")
    
    if category_l2.lower() not in l2_categories:
        if language.value == "english":
            return {
                "status": 432,
                'response': "undefined level 2 category",
                "message": chat_resp
                }
        
        to_translate = ["status", 432, "response", "undefined level 2 category", "message", chat_resp]
        
        result = get_translation(to_translate, language.value)
        return result

    # Ticket generation
    ticket_chain = LLMChain(llm=llm, prompt=ticket_prompt)
    ticket_result = ticket_chain.run(examples=TICKET_DATA, user_input=context)
    
    logger.info("Complaint ticket classification completed")

    # # Response generation based on language choice
    # response_eg = get_response_eg(response_data, category_l2)
    # response_chain = LLMChain(llm=llm, prompt=response_prompt)
    # response = response_chain.run(examples=response_eg, user_input=context)
    
    logger.info("Complaint Response classification completed")
    
    if language.value == "english":
        return {
            "status": 200,
            "complaint_summary":context,
            "ticket status": ticket_result,
            "Level 1 Category": category_l1,
            "category": category_l2,
            # "response": response
        }
    
    to_translate = [
        "status", 200,
        "complaint summary", context,
        "ticket status", ticket_result,
        "Level 1 Category", category_l1,
        "category", category_l2,
        # "response", response
        ]
    
    result = get_translation(to_translate, language.value)
    
    logger.info("Result Translation completed")
    
    return result