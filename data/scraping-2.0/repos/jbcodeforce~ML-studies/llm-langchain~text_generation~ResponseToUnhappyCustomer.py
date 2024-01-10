import os
import sys
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from bedrock.utils import bedrock, print_ww

'''
Generate an email response to a customer who was not happy with the quality of customer service 
that they received from the customer support engineer. We will provide additional context to the model 
by providing the contents of the actual email that was received from the unhappy customer.
'''
bedrock_runtime = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}


textgen_llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_runtime,
    model_kwargs=inference_modifier,
)

'''
By creating a template for the prompt we can pass it different input variables to it on every run. 
This is useful when we have to generate content with different input variables that we may be 
fetching from a database. 
'''
multi_var_prompt = PromptTemplate(
    input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"], 
    template="""

Human: Create an apology email from the Service Manager {customerServiceManager} to {customerName} in response to the following feedback that was received from the customer: 
<customer_feedback>
{feedbackFromCustomer}
</customer_feedback>

Assistant:"""
)

prompt = multi_var_prompt.format(customerServiceManager="Bob", 
                                 customerName="John Doe", 
                                 feedbackFromCustomer="""Hello Bob,
     I am very disappointed with the recent experience I had when I called your customer support.
     I was expecting an immediate call back but it took three days for us to get a call back.
     The first suggestion to fix the problem was incorrect. Ultimately the problem was fixed after three days.
     We are very unhappy with the response provided and may consider taking our business elsewhere.
     """
     )
num_tokens = textgen_llm.get_num_tokens(prompt)
print(f"Our prompt has {num_tokens} tokens")
response=textgen_llm(prompt)
email = response[response.index('\n')+1:]
print_ww(email)


