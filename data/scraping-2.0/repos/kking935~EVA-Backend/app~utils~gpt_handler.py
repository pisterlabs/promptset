'''
This file handles all interactions with the OpenAI API,
including the prompting, decoding of responses, and building the report.
'''
import os
from dotenv import load_dotenv
import openai

from app.models import SurveyModel, SurveyQuestion

from ..config.sdoh_domains import SDOH_DOMAINS
from ..config.gpt_settings import INITIAL_SYSTEM_PROMPT, INITIAL_QA_EXAMPLES, MODEL, MAX_TOKENS, TEMPERATURE

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def handle_new_messages(messages):
    response_data = openai.ChatCompletion.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=messages,
    )
    response_message = response_data.choices[0].message.content
    messages += [{"role": "assistant", "content": response_message}]
    return response_message

def create_system_prompt():
    prompt = INITIAL_SYSTEM_PROMPT
    for domain_id, domain_obj in SDOH_DOMAINS.items():
        # prompt += f"{domain_obj['domain']} ({domain_id}):"
        prompt += f"{domain_obj['domain']}:"

        for subdomain_id, subdomain_obj in domain_obj['subdomains'].items():
            # prompt += f" {subdomain_obj['subdomain']} ({subdomain_id}),"
            prompt += f" {subdomain_obj['subdomain']},"

        prompt = prompt[:-1] + '. ' # remove trailing comma and add period
    return prompt[:-1] # remove trailing space

def build_initial_messages():
    initial_messages = [{"role": "system", "content": create_system_prompt()}]
    for qa_example in INITIAL_QA_EXAMPLES:
        for prompt in qa_example['user_prompts']:
            initial_messages.append({"role": "user", "content": prompt})
        initial_messages.append({"role": "assistant", "content": qa_example['assistant_response']})
    return initial_messages

def decode_risk_factors(response):
    risk_factors = {}
    domains = response.split(".")
    for domain in domains:
        domain_subdomains = domain.split(":")
        domain = domain_subdomains[0].strip()
        if len(domain_subdomains) < 2:
            continue
        if not domain in risk_factors:
            risk_factors[domain] = set()
        subdomains = [subdomain.strip() for subdomain in domain_subdomains[1].split(",")]
        for subdomain in subdomains:
            if not subdomain in risk_factors[domain]:
                risk_factors[domain].add(subdomain)
    return risk_factors

# def build_risk_factors(survey, messages):
#     overall_risk_factors = {}
#     for question_obj in survey.values():
#         messages += [
#             {"role": "user", "content": f"Question {question_obj.qid}: {question_obj['question']}"}, 
#             {"role": "user", "content": f"Answer {question_obj.qid}: {question_obj['answer']}"}
#         ]
#         response_message = handle_new_messages(messages)
#         risk_factors = decode_risk_factors(response_message)
#         question_obj['risk_factors'] = risk_factors
#         overall_risk_factors = {key: overall_risk_factors.get(key, set()) | risk_factors.get(key, set()) for key in set(overall_risk_factors.keys()) | set(risk_factors.keys())}
#     return overall_risk_factors

def generate_individual_risks(survey: SurveyModel, survey_question: SurveyQuestion):
    survey.messages += [
        {"role": "user", "content": f"Question {survey_question.qid}: {survey_question.question}"}, 
        {"role": "user", "content": f"Answer {survey_question.qid}: {survey_question.answer}"}
    ]
    response_message = handle_new_messages(survey.messages)
    risk_factors = decode_risk_factors(response_message)
    survey_question.risk_factors = risk_factors
    print(risk_factors)
    survey.overall_risk_factors = {key: survey.overall_risk_factors.get(key, set()) | risk_factors.get(key, set()) for key in set(survey.overall_risk_factors.keys()) | set(risk_factors.keys())}

def initialize_survey(survey):
    survey.messages = build_initial_messages()
    survey.overall_risk_factors = {}
    survey.summary = ""
    return survey

def build_summary(messages):
    messages += [
        {"role": "user", "content": f"Summarize the patient's key social risk factors with direct references from their answers."}, 
    ]
    summary = handle_new_messages(messages)
    return summary

def finalize_survey(survey: SurveyModel):
    survey.summary = build_summary(survey.messages)
    return survey