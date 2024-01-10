import json
import logging

from council.agent_tests import AgentTestSuite, AgentTestCase
from council.agents import Agent
from council.chains import Chain
from council.contexts import AgentContext
from council.contexts import Budget
from council.controllers import LLMController
from council.evaluators import LLMEvaluator
from council.filters import BasicFilter
from council.scorers import LLMSimilarityScorer
from council.skills import LLMSkill
from council.llm import OpenAILLM
import dotenv

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(threadName)s $(name)s:#(funcName)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m_d %H:%M:%S%z",
)

logging.getLogger("council").setLevel(logging.DEBUG)

dotenv.load_dotenv()

openai_llm = OpenAILLM.from_env()

finance_prompt = "You are an assistant expert in Finance. When asked about something else, say you don't know"
finance_skill = LLMSkill(llm=openai_llm, system_prompt=finance_prompt)
finance_chain = Chain(name="finance", description="Answers questions about finance", runners=[finance_skill])

game_prompt = "You are an expert in video games. When asked about something else, say you don't know"
game_skill = LLMSkill(llm=openai_llm, system_prompt=game_prompt)
game_chain = Chain(name="game", description="Answers questions about video games", runners=[game_skill])

fake_prompt = "You will provide an answer not related to the question"
fake_skill = LLMSkill(llm=openai_llm, system_prompt=fake_prompt)
fake_chain = Chain(name="fake", description="Can answer all questions", runners=[fake_skill])

controller = LLMController(chains=[finance_chain, game_chain, fake_chain], llm=openai_llm, top_k=2)
evaluator = LLMEvaluator(llm=openai_llm)
agent = Agent(controller=controller, evaluator=evaluator, filter=BasicFilter())

context = AgentContext.from_user_message(message="what is inflation?", budget=Budget(40))
#context = AgentContext.from_user_message(message="what are the most popular video games?", budget=Budget(20))
#fake_context = AgentContext.from_user_message(message="what is the age of the captain?", budget=Budget(20))

results = agent.execute(context=context)

print("Response from finance agent:")
for item in results.messages:
    print("----")
    print(f"score: {item.score}")
    print(item.message.message)

tests = [
    AgentTestCase(
        prompt="What is inflation?",
        scorers=[
            LLMSimilarityScorer(
                llm=openai_llm,
                expected="Inflation is the rate at which the general level of prices for goods and services is rising,\
                 and, subsequently, purchasing power is falling",
            )
        ],
    ),
    AgentTestCase(
        prompt="What are the most popular video games?",
        scorers=[
            LLMSimilarityScorer(
                llm=openai_llm,
                expected="The most popular video games are: ...",
            )
        ],
    ),
    AgentTestCase(
        prompt="What are the most popular movies?",
        scorers=[
            LLMSimilarityScorer(
                llm=openai_llm,
                expected="The most popular movies are ...",
            )
        ],
    )
]

suite = AgentTestSuite(test_cases=tests)
result = suite.run(agent=agent)

print(json.dumps(result.to_dict(), indent=2))