'''
RBI - Research, backtest, implement
LLMs - RBI 

git: https://github.com/moondevonyt/AI---LLMs-For-Automated-Trading
'''
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
import langchainkeys as l 
import time 

###### RBI SYSTEM ########

#### RESEARCH LLM ########
# Research trading strategies
llm_research = OpenAI(openai_api_key=l.open_ai_key, temperature=0)
research_template = """look at the recent market data for bitcoin and make a trading strategy for it. use {indicator} of your choice. over the last {time_period} days """
research_prompt = PromptTemplate(template=research_template, input_variables=["indicator", "time_period"])
research_chain = LLMChain(prompt=research_prompt, llm=llm_research)

# Generate trading strategies 
research_result = research_chain.run({"indicator": "any indicator", "time_period": "365"})

# Retrieve the generated text
trading_strategy = research_result

print(trading_strategy)
print('')
print('done thinking of strategies... moving on to instructions for backtests...')
print('')
time.sleep(5)

#### STRATEGY INSTRUCTIONS LLM ########
# Generate step-by-step instructions for the trading strategy
llm_instructions = OpenAI(openai_api_key=l.open_ai_key, temperature=0)
instructions_template = """
Based on the generated trading strategy:
- Determine the entry condition.
- Define the exit condition.
- Specify the market stay-out condition.

Trading Strategy:
{trading_strategy}

Entry Instructions:
...

Exit Instructions:
...

Market Stay-out Instructions:
...
"""
instructions_prompt = PromptTemplate(template=instructions_template, input_variables=["trading_strategy"])
instructions_chain = LLMChain(prompt=instructions_prompt, llm=llm_instructions)

print('made it to line 53')
# Generate instructions for the trading strategy
instructions_result = instructions_chain.run({"trading_strategy": trading_strategy})
step_by_step_outlines = [instructions_result]
print(step_by_step_outlines)

# Print the step-by-step outline
for outline in step_by_step_outlines:
    print(outline)

# Print completion message
print("\nAll done with the research and step-by-step instructions!")

time.sleep(8756) # to not go passed

#### BACKTEST LLM ########
# use the ideas that the LLM above came up with and build a backtest
backtesting_strategies = research_result.output 

# implement the backtesing logic 
#TODO - #backtestint_code =

#### BUG TESTING LLM ######
llm_debugging = OpenAI(openai_api_key=l.open_ai_key, temperature=0)

# define a prompt template for code debugging
debugging_template = """give the backtesting coe, identify and fix any coding bugs or issues"""

debugging_prompt = PromptTemplate(template=debugging_template)

# create chain for code debugging
debugging_chain = LLMChain(prompt=debugging_prompt, llm=llm_debugging)

debugging_result = debugging_chain.run(backesting_code)

# fix any coding bugs in the backtesting code
fixed_backtesting_code = debugging_result.output 