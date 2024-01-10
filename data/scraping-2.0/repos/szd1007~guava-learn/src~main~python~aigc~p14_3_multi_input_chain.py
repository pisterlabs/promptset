import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import SequentialChain

openai.api_key = os.environ.get("OPENAI_API_KEY")


#https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
llm = OpenAI(model_name="gpt-4-1106-preview", max_tokens=2048, temperature=0.5)

q1_prompt = PromptTemplate(
    template="{year1}年的欧冠联赛的冠军是哪支球队，只说球队名称。", input_variables=["year1"]
)

q2_prompt = PromptTemplate(
    template="{year2}年的欧冠联赛的冠军是哪支球队，只说球队名称。", input_variables=["year2"]
)

q3_prompt = PromptTemplate(
    template="{team1}和{team2}哪只球队获得欧冠的次数多一些？", input_variables=["year1", "year2"]
)
chain1 = LLMChain(llm=llm, prompt=q1_prompt, output_key="team1")
chain2 = LLMChain(llm=llm, prompt=q2_prompt, output_key="team2")
chain3 = LLMChain(llm=llm, prompt=q3_prompt)

answer_chanin = SequentialChain(
    chains=[chain2, chain1, chain3], input_variables=["year1", "year2"],  verbose=True
)
answer = answer_chanin.run(year1=2000, year2=2010)

print(answer)