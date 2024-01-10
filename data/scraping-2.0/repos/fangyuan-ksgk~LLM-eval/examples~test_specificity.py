import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arthur_bench.run.testsuite import TestSuite


print('Verification of APIKEY: ', os.environ['OPENAI_API_KEY'])

from openai import OpenAI
client = OpenAI()
try:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":"You are a poetic assistant, skilled in explaining stuff in a poetic way."},
            {"role": "user", "content":"What is the meaning of life?"},
        ]
    )
except:
    raise RuntimeError(f"Please set OPENAI_API_KEY into sys environment variable")

print('OPENAI completion check: ')
print(completion.choices[0].message)


eli5 = pd.read_csv('./specificity/eli5_25.csv')

# TestSuite contains scorer & question & context
suite_spec = TestSuite(
    name='specificity',
    scoring_method='specificity',
    reference_data=eli5,
    input_column='history'
)

# run test 
# Question: What are we testing here? The question, the candidate answers and the ratings are all in the dataframe. 
# Since we are not trying to train anything here, we must be aiming for something missing from the dataframe?
# Or are we just trying to mask the score, and compare the predicted score with the actual score? -- that is possible
run_A = suite_spec.run(
    run_name="A",
    candidate_data=eli5,
    candidate_column='human_ref_A',
    replace_existing=True
)

run_B = suite_spec.run(
    run_name="B",
    candidate_data=eli5,
    candidate_column='human_ref_B',
    replace_existing=True
)

A_scores = []
for t in run_A.test_cases:
    A_scores.append(t.score)
B_scores = []
for t in run_B.test_cases:
    B_scores.append(t.score)
print('A scores: ', A_scores, ' | length: ', len(A_scores))
print('B scores: ', B_scores, ' | length: ', len(B_scores))


# plot : nice ! although distplot seems to be deprecated now
# sns.set_style("whitegrid")
# sns.set_palette("husl")
# plt.figure(figsize=(10, 6))
# sns.distplot(A_scores, label='A')
# sns.distplot(B_scores, label='B')
# plt.legend()
# plt.title('Distribution of scores')
# plt.xlabel('Score')
# plt.ylabel('Density')
# plt.show()


from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
import openai
import time
from tqdm import tqdm

print('Langchain import complete')
print('-'*80)

#prompt - from https://github.com/i-Eval/FairEval/blob/main/FairEval.py

system_message_prompt = SystemMessagePromptTemplate.from_template(
  "You are a helpful and precise assistant for checking the helpfulness of an answer to a specific prompt."
  """We would like to request your feedback on the helpfulness of 2 responses to the PROMPT.
    Please rate the helpfulness, as measured by how useful, relevant and the level of details of the responses.

    Each response receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
    Then, output two lines indicating the scores for Response 1 and 2, respectively.

    Output with the following format:

    Evaluation evidence: <your evluation explanation here>
    Score of the Response 1: <score>
    Score of the Response 2: <score>"""
) # Fundamentally we wish to compare the two responses, and we want to know which is better. @LLM

# print('Type of System message prompt: ', type(system_message_prompt))

comparison_template = HumanMessagePromptTemplate.from_template(
    """
    PROMPT: {prompt}
    Response 1: {response_1}
    The end of Response 1.
    ----
    Response 2: {response_2}
    The end of Response 2.
    """
) # This we fill from the dataframe @Dataset

llm_evaluate = ChatPromptTemplate.from_messages([system_message_prompt, comparison_template])

# print('LLM Evaluate: ', llm_evaluate)

llmchain= LLMChain(llm=ChatOpenAI(temperature=0, max_tokens=512), prompt=llm_evaluate)

# Calling LLM here -- query model
def query(prompt, response_1, response_2):
    for i in range(4): #max API RETRY
        try:
            # basically dictionary of {} inputs to fill the templates and form a prompt
            output = llmchain({"prompt": prompt, "response_1": response_1, "response_2": response_2})
            print('Output Keys: ', output.keys())
            response = output["text"]
            return response
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(30)
        except Exception as e:
            print('error')
    raise RuntimeError(f"Failed after 4 retries.")

def parse_score_from_review(review):
    try:
        score1 = review.split("\n")[-2]
        score2 = review.split("\n")[-1]
        score1 = score1.split(":")[-1].strip() # remove content inside the brackets, in this case, whitespace is removed
        score2 = score2.split(":")[-1].strip()
        return [float(score1), float(score2)]
    except:
        print(f'Failed to parse scores from {review}')
        return [-1, -1]

def get_scores(prompts, responses_1, responses_2):
    llm_score_1= []
    llm_score_2= []
    all_scores=[]
    for i in tqdm(range(len(prompts))):
        # we are permuting order here, and expecting different scores?
        # I thought we explicitly put 'permutation-invariant please' in the prompt
        # then if you would just paraphrase it a little bit, you definitely would get a different score ...
        score_1a, score_2a= parse_score_from_review(query(prompts[i], responses_1[i], responses_2[i]))
        score_2b, score_1b = parse_score_from_review(query(prompts[i], responses_2[i], responses_1[i]))
        all_scores.append([score_1a, score_1b, score_2a, score_2b])
        score_1 = (score_1a + score_1b)/2
        score_2 = (score_2a + score_2b)/2
        llm_score_1.append(score_1)
        llm_score_2.append(score_2)
    return llm_score_1, llm_score_2, all_scores



p= eli5['history'].values
r1=eli5['human_ref_A'].values
r2=eli5['human_ref_B'].values


llm_score_1, llm_score_2, all_scores = get_scores(p, r1, r2)


print('LLM Score 1: ', llm_score_1)
print('LLM Score 2: ', llm_score_2)




