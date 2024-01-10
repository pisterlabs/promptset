# flake8: noqa
from langchain.prompts import PromptTemplate

template = """You are a Sales Manager scoring the work of your direct reports for their performance review.
Your reports send cold emails to potential customers. 
You are given a JSON object containing data about the customer, and two emails: one sent by SDR A, and one sent by SDR B.
Choose the better email between EMAIL A and EMAIL B, and provide an explanation.

Example Format:
CUSTOMER: JSON object here
EMAIL A: the email written by the SDR A
EMAIL B: the email written by the SDR B
CHOICE: EMAIL A or EMAIL B. Provide an explanation why this was the better choice.

Review the work based on whether or not the SDR email correctly addressses the customer, is clear, and convincing.

CUSTOMER: {customer}
EMAIL A: {email_a}
EMAIL B: {email_b}
CHOICE:"""
CHOICE_PROMPT = PromptTemplate(
    input_variables=["customer", "email_a", "email_b"], template=template
)