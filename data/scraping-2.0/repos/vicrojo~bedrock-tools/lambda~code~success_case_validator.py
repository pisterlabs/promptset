import json
import boto3
import os
from langchain import PromptTemplate
from langchain.llms import Bedrock
from langchain.chains import LLMChain

def evaluate_public_case (public_case_text):
    claude = Bedrock(
        model_id="anthropic.claude-v1",
    )
    claude.model_kwargs = {'temperature': 0.3, 'max_tokens_to_sample': 4096}
    
    template = """
    You are an AWS evaluator and validator of case studies submitted by AWS partners. 
    You have defined the following criteria to evaluate each case study:
    
    1. Must focus on Conversational AI or related technologies (for example chatbots, conversations, voicebots, copilots, virtual assistants, etc.)
    2. Must be in production, not pilot or proof of concept stages.
    3. Must include a paragraph about the customer's business, industry, and name.
    4. Must describe the customer's business challenge or IT need and potential consequences if not addressed.
    5. Must include references to AWS technologies or how they were leveraged
    6. Must describe the outcome related to the initial challenge and provide specific metrics about the benefits.
    7. Must include a paragraph about the partner's business, industry, APN programs, tier level, and type.
    8. Must not require authentication or registration to access its details
    9. Must be less than 18 months old
    10 Partners cannot use themselves or affiliate companies as references
    11 Customers cannot be used multiple times
    12 PR statements cannot be used as public case studies
    13 Information must not needed registration or authentication to access it
    
    Evaluate the following case study information:
    {case_study}
    
    Breakdown the criteria not met by the case study, with a short explanation.
    answer:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["case_study"],
        template=template
    )
    
    llm_chain = LLMChain(
        llm=claude, verbose=True, prompt=prompt_template
    )
    
    results = llm_chain(public_case_text)
    return results

def lambda_handler(event, context):
    case_study = event["case_study"]
    bedrock = boto3.client('bedrock') 
    case_study_result = evaluate_public_case(case_study)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/plain'
        },
        'case_results': case_study_result["text"]
    }