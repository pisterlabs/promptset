import jsonlines
from langchain.schema import SystemMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from dotenv import load_dotenv
load_dotenv()
import re


search = SerpAPIWrapper()
# obs = search.results(action_input)
# print(obs["answer_box"]["link"])


#    - Beware of unused EC2 instance time, if EC2 instances are only used for a few minutes every hour, then use AWS Lambda instead, you pay only  for the compute time, not for unused time when EC2 instance is active
#     - A single Kinesis stream shard has a quota of 1 MB/s
#     - credentials that are supplied by  AWS Single Sign -On (AWS SSO ) are temporary
#     - AWS CloudFormation StackSets  can deploy the IAM role across multiple accounts with a single operation
#     - When granting permissions to a third-party monitoring solution, it is important to ensure that the solution has the required permissions across all AWS accounts. This can be done by creating an IAM role in each account and granting the third-party monitoring solution access to the role. Additionally, the trust policy of the IAM role should specify the AWS account of the third-party monitoring solution.
  
  
# - By using AWS Organizations, the management account would have control over the other accounts
#     - By setting up billing alerts and CloudWatch alarms, the account administrator will be notified when the account exceeds a designated spending threshold, allowing them to take action to prevent excessive spending. This solution also allows each business group to retain full control of its AWS account. In general, setting up billing alerts and CloudWatch alarms is a good practice to ensure that spending does not exceed the desired amount.
#     - CORS must be enabled in API Gateway, not in the S3 bucket. Cross-origin resource sharing (CORS) is a mechanism that allows restricted resources on a web page to be requested from another domain outside the domain from which the resource originated. In order for the form to successfully post to the API endpoint and receive a valid response, CORS must be enabled in API Gateway.
#     - S3 bucket must be configured for web hosting and CORS must be enabled in API Gateway. Web hosting on S3 allows the form to be hosted in a public S3 bucket, and CORS must be enabled in API Gateway in order for the form to successfully post to the API endpoint and receive a valid response. In general, when building an HTML form that is hosted in a public S3 bucket and uses JavaScript to post data to an API endpoint, the S3 bucket must be configured for web hosting and CORS must be enabled in API Gateway.
      

prompt = PromptTemplate(template="""
    You are an AWS Architect expert. You understand AWS cloud architecture and can give architecture recommendations.
    Here are some background knowledge that can be useful for you.

    - Create an IAM role in the organization's management account. Allow the AWS account of the third-party monitoring solution to assume the role does not provide the third-party monitoring solution with the required permissions across all AWS accounts. It only provides the third-party monitoring solution with permissions in the organization's management account.
    - creates an IAM role in the organization's management account that allows the AWS account of the third-party monitoring solution to assume the role. This IAM role is then created across all linked AWS accounts by using a stack set, providing the third-party monitoring solution with the required permissions across all AWS accounts. In general, when granting access to multiple AWS accounts, it is best practice to use a stack set to create the same IAM role across all accounts. This ensures that the same permissions are applied to all accounts, making it easier to manage and maintain.
    
    Please select the appropriate answer given the following questions as best as you can.
    
    Use the following format:
    
    Question: the input question you must answer with the multiple choices to choose from
    Thought: You should always explain your thought process to the humans
    Is the question asking for single answer or multiple answer? Single or Multiple
    Answer: The choice or choices you pick from the provided multiple choices, only provide the answer label(s) like A, B, C, D, E or F.
    
    Begin!
    
    Question: {question}
    Thought: """.strip(), 
    input_variables=["question"])
llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)



critic_prompt = PromptTemplate(template="""
You are an AWS Architect expert. You understand AWS cloud architecture and can give architecture recommendations.

Given the question {question}

And the corresponding answer {answer}

Please explain why the answer {wrong_answer} is wrong and summarize it into general knowledge that is not only specific to this question, and then explain why the answer {answer} is correct and summarize it into general knowledge that is not only specific to this question.
""", input_variables=["question", "answer", "wrong_answer"])
llm_critic_chain = LLMChain(llm=llm, prompt=critic_prompt)



summarize_critic_prompt = PromptTemplate(template="""
You are an AWS Architect expert. You understand AWS cloud architecture and can give architecture recommendations.

Given the following sentence, please remove the parts of it that is too specific to a question {sentence}                                
""", input_variables=["sentence"])
llm_summarize_critic_chain = LLMChain(llm=llm, prompt=summarize_critic_prompt)



with jsonlines.open("data/prod/train.jsonl", "r") as f:
    with jsonlines.open("data/prod/eval.jsonl", "w") as evalf:
        for item in f:
            question = item["question"].strip()
            
            if item['id'] == 1:
                llm_output = llm_chain.run(question=question)
                llm_answer_choices = llm_output[llm_output.index("Answer: ") + 8 : ].strip().split(",")
                llm_answer_choices = [a.strip() for a in llm_answer_choices]
                # evalf.write({"id" : item["id"], "is_correct" : set(llm_answer_choices) == set(item["answer_choice"]), 
                #             "llm_output" : llm_output, "correct_choices" : item["answer_choice"]})
                
                
                print(item["id"], llm_answer_choices, item["answer_choice"])
                is_correct = set(llm_answer_choices) == set(item["answer_choice"])
                
                if not is_correct:
                    critic = llm_critic_chain.run(question=question, answer=", ".join(item["answer_choice"]), wrong_answer=", ".join(llm_answer_choices))
                    print(item["id"], question)
                    print("CRITIC", critic)
            
            # out = llm_summarize_critic_chain.run(sentence=critic)
            # print(out)
    