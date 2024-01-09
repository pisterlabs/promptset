from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chains import PALChain
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

from ogbujipt.config import openai_emulation
from ogbujipt.model_style.alpaca import prep_instru_inputs, ALPACA_PROMPT_TMPL

from langchain.prompts import PromptTemplate
openai_emulation(host="http://192.168.0.73", port="8000")

def simpleWordPrompt():
    prompt = PromptTemplate(
        input_variables=["place"],
        template="What is the capital of {place}?",
    )
    print(prompt.format(place="Nigeria"))

    llm = OpenAI(temperature=0.1)
    llmchain = LLMChain(llm=llm, prompt=prompt)
    response = llmchain.run(place="Nigeria")
    print(response)


def MathWorldProblem():
    llm = OpenAI(temperature=0.1)
    palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
    response = palchain.run(
        "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"
        )
    print(response)


def agentTest():
    llm = OpenAI(temperature=0)
    tools = load_tools(["pal-math"], llm=llm)
    agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)
    agent.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")


def main():
    MathWorldProblem()


if __name__ == "__main__":
    main()