r"""°°°
# Getting Started
°°°"""
# |%%--%%| <gOPXQazOh0|yMwLn4kyVM>

from langchain.llms import OpenAI

# n: how many completions to generate for each prompt
llm = OpenAI(model_name='text-ada-001', n=2, best_of=2, temperature=1)
llm("tell me a joke !")


#|%%--%%| <yMwLn4kyVM|5F2CisYISi>
r"""°°°
you can call it with a list of inputs, getting back a more complete response than just the text. This complete response includes things like multiple top responses, as well as LLM provider specific information.
°°°"""
#|%%--%%| <5F2CisYISi|4hSGpG9eHG>

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 15)

#|%%--%%| <4hSGpG9eHG|kZMW1cS1Qk>

len(llm_result.generations)

#llm_result.to_dict() # see result details

print(llm_result.generations[0])
print("\n\n")
print(llm_result.generations[-1])
llm_result.llm_output


#|%%--%%| <kZMW1cS1Qk|1Kg9Ct1muS>
r"""°°°
## estimate number of tokens in prompt
°°°"""
#|%%--%%| <1Kg9Ct1muS|qz7lnBufdW>

llm.get_num_tokens("what is a joke")

#|%%--%%| <qz7lnBufdW|Q9fgYuBKEK>
r"""°°°
# Key Concepts

- Core method exposed by llms is `generate`: takes list of str returns LLMResult
- Can also be called directly with single string as input and returns a stirng
- Main result is `LLMResult`, input list of strings -> list of LLMResult
  Each result is a list of generations (since you can request `n` generations per input str)
- `llm_output` contains provider specific ouptput
°°°"""
#|%%--%%| <Q9fgYuBKEK|3k0hrhqP7F>
r"""°°°
## LLM Serialization

Wrinting and reading llms to disk
°°°"""
#|%%--%%| <3k0hrhqP7F|9V7QPvLLcT>

from langchain.llms.loading import load_llm

llm.save("llm.json")

#|%%--%%| <9V7QPvLLcT|rcdMSwUd3W>

llm = load_llm("llm.json")

#|%%--%%| <rcdMSwUd3W|JfkBS05EUP>
r"""°°°
## Token Usage Tracking


°°°"""
#|%%--%%| <JfkBS05EUP|FUnKToXsu6>

from langchain.callbacks import get_openai_callback

#|%%--%%| <FUnKToXsu6|9AoUdMfzg7>

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

#|%%--%%| <9AoUdMfzg7|aiTKa2iDUx>

with get_openai_callback() as cb:
    result = llm("tell me a joke")
    print(cb.total_tokens)

#|%%--%%| <aiTKa2iDUx|6G5rwEeItx>
r"""°°°
Anything inside the context manager will get tracked.

Example tracking multiple calls
°°°"""
#|%%--%%| <6G5rwEeItx|uDJtapBCby>

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    result2 = llm("Tell me a funny joke")
    print(cb.total_tokens)

#|%%--%%| <uDJtapBCby|sLuvD8dXnk>
r"""°°°
If a chain or agent with multiple steps in it is used, it will track all those steps.
°°°"""
#|%%--%%| <sLuvD8dXnk|uUQOyO8XIw>

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["searx-search", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

#|%%--%%| <uUQOyO8XIw|nv0tixnGfg>

with get_openai_callback() as cb:
    res = agent.run("What is the temperature in Paris  Berlin and Granada ?  \
                    Print every city's temperature in Celcius and Fahrenheit. Think step by step")

#|%%--%%| <nv0tixnGfg|vpDGqKnagk>

    print(cb.total_tokens)

# |%%--%%| <vpDGqKnagk|xUoMvs6XZ0>


