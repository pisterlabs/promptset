import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

bedrock_client = boto3.client("bedrock-runtime")
model_name = "anthropic.claude-v2"

claude = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_client,
    model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 4096},
)


template = """
Human: Answer the following questions as best you can. You have access to the following tools:
Search: A search engine. Useful for when you need to answer questions about current events. Input should be a search query.
Use the following instructions:
1. Question: the input question you must answer
2. Thought: you should always think about what to do
3. Action: the action to take, should be one of [Search]
4. Action Input: the input to the action
5. Observation: the result of the action
6. Thought: I now know the final answer
7. Final Answer: the final answer to the original input question
The steps 1 to 5 can repeat N times
After you answer my following question, I want you to try and verify the answer.
The verification process is as follows:
Step 1: Examine the answer and identify elements that might be important to verify, such as notable facts, figures, and any other significant considerations. 
Step 2: Come up with verification questions that are specific to those identified elements. 
Step 3: Separately answer each of the verification questions, one at a time. 
Step 4: Finally, after having answered the verification questions, review the initial answer that you gave to my question and adjust the initial answer based on the results of the verification questions. 
Other aspects: Make sure to show me the verification questions that you come up with, and their answers, and whatever adjustments to the initial answer you are going to make. It is okay for you to make the adjustments and you do not need to wait for my approval to do so. Do you understand all of these instructions?
Return only your answer to the human question as a JSON object in form of key:value pairs. The final answer must be named with the key "FinalAnswer", and break it down into the different names and values that compose it, in the form of key-value pairs too.
what is the weather where in {place}?

Assistant:
"""

prompt_template = PromptTemplate(
    input_variables=["place"],
    template=template
)

llm_chain = LLMChain(
    llm=claude, verbose=True, prompt=prompt_template
)

place = "Las Vegas"
results = llm_chain.predict(place=place)

print(results)