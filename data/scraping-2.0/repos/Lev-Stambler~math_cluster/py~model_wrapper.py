import os, sys
from langchain.llms import fake
from langchain.llms import DeepInfra
from langchain import OpenAI, PromptTemplate, LLMChain

USE_FAKE = False
# if not USE_FAKE:
#     sys.path.append(os.path.join(os.getcwd(), "exllama"))
#     import exllama_lang


# get a new token: https://deepinfra.com/login?from=%2Fdash

# from getpass import getpass

# DEEPINFRA_API_TOKEN = getpass()
# os.environ["DEEPINFRA_API_TOKEN"] = DEEPINFRA_API_TOKEN




if USE_FAKE:
    llm_name = "fake"
    llm = fake.FakeListLLM(responses=["hello " * 30] * 1_000)
else:
		# llm = OpenAI(model_name="gpt-3.5-turbo-16k")
		# llm_name = "gpt-3.5-turbo-16k"

		llm_name = "airborors-llama-2"
		llm = DeepInfra(model_id="jondurbin/airoboros-l2-70b-gpt4-1.4.1")
		llm.model_kwargs = {
		    "temperature": 0.2,
		    "repetition_penalty": 1.2,
		    "max_new_tokens": 250,
		    "top_p": 0.9,
		}

    # llm = exllama_lang.ExLLamaLLM(model_dir="../../Luban-13B-GPTQ", max_response_tokens=1_000,
    #                               max_seq_len=4_096, temperature=0.3, beams=3, beam_length=10)