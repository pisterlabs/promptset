#refer:https://python.langchain.com/docs/integrations/providers/openllm
#https://python.langchain.com/docs/integrations/llms/openllm
import os
from openai import OpenAI
from langchain.llms import OpenLLM
from langchain.adapters import openai as lc_openai
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults

# python3 localLLM/testOpenLLM.py
from langchain.adapters import openai as lc_openai
# Do this so we can see exactly what's going on under the hood
from langchain.globals import set_debug
from langchain.globals import set_verbose
from getpass import getpass

#for debuging refer:https://python.langchain.com/docs/guides/debugging
set_debug(True)
set_verbose(True)
print("==============================================Completed the setup env=========================")
#==============================================Completed the setup env=========================




#==============================================exact logic code with openAI version higher 1.0.0=========================
#Initialize LLM language model
base_url = os.environ.get("TEST_API_BASE_URL", "http://192.168.0.232:1234")     # 1234  3000
api_key = os.environ.get("OPENAI_API_KEY", "xxxxxxxxx")   # even your local don't use the authorization, but you need to fill something, otherwise will be get exception.
api_key = "xxxx"

#Wrapper for OpenLLM server
llm = OpenLLM(server_url=base_url)

# #Wrapper for Local Inference. PS: This got "OSError: [Errno 28] No space left on device" in my local
# #llm = OpenLLM(model_name="dolly-v2", model_id='databricks/dolly-v2-7b')
# model_id = 'databricks/dolly-v2-7b'  # databricks/dolly-v2-3b
# llm = OpenLLM(
#     model_name="dolly-v2",
#     model_id="databricks/dolly-v2-3b",
#     temperature=0.94,
#     repetition_penalty=1.2,
# )


#llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
#llm.invoke("Write a ballad about LangChain")
# print("ChatGoogleGenerativeAI.result=>", result)

#Integrate with a LLMChain  refer:https://python.langchain.com/docs/integrations/llms/openllm
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
generated = llm_chain.run(product="mechanical keyboard")
print(generated)



'''
#Based OpenLLM class(langchain.llms) run on openLLM server(port=3000) Always got exception with calling Wrapper for OpenLLM server, Also OpenLLM CLI and curl+http also not working for OpenLLM,Even I changed the LLM model to LLM model=flan-t5
Traceback (most recent call last):
  File "/Users/ming/Documents/work/repo/tmp/tmp/llm_sample/langChain/localLLM/testOpenLLM.py", line 40, in <module>
    llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 892, in __call__
    self.generate(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 635, in generate
    params = self.dict()
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 990, in dict
    starter_dict = dict(self._identifying_params)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_community/llms/openllm.py", line 220, in _identifying_params
    self.llm_kwargs.update(self._client._config())
TypeError: 'dict' object is not callable



#Based OpenLLM class(langchain.llms) can't run with LM studio server(port=1234) got the exception. Even I used openllm CLI client connect to LM studio server(port=1234) got same error.
#Summary => Seem like something different between OpenLLM class(langchain.llms) SDK and official openai SDK.
(minglocalenv3.9) ming@Mings-MacBook-Pro langChain % python3 localLLM/testOpenLLM.py
[chain/start] [1:chain:LLMChain] Entering Chain run with input:
{
  "product": "mechanical keyboard"
}
=========ming start=>self._client._config()
<HTTPClient address=http://192.168.0.232:1234 timeout=Timeout(timeout=30) api_version=v1 verify=True>
[chain/error] [1:chain:LLMChain] [386ms] Chain run errored with error:
"ValueError(KeyError('configuration'))Traceback (most recent call last):\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/base.py\", line 306, in __call__\n    self._call(inputs, run_manager=run_manager)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/llm.py\", line 103, in _call\n    response = self.generate([inputs], run_manager=run_manager)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/llm.py\", line 115, in generate\n    return self.llm.generate_prompt(\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py\", line 516, in generate_prompt\n    return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py\", line 635, in generate\n    params = self.dict()\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py\", line 990, in dict\n    starter_dict = dict(self._identifying_params)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_community/llms/openllm.py\", line 222, in _identifying_params\n    self.llm_kwargs.update(self._client._config())\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_http.py\", line 66, in _config\n    self.__config = self._metadata.configuration\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_http.py\", line 60, in _metadata\n    self.__metadata = self._post(path, response_cls=Metadata, json={}, options={'max_retries': self._max_retries})\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py\", line 445, in _post\n    return self.request(response_cls, RequestOptions(method='POST', url=path, json=json, **options), stream=stream, stream_cls=stream_cls)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py\", line 369, in request\n    return self._request(response_cls=response_cls, options=options, remaining_retries=remaining_retries, stream=stream, stream_cls=stream_cls)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py\", line 401, in _request\n    return self._process_response(response_cls=response_cls, options=options, raw_response=response, stream=stream, stream_cls=stream_cls)\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py\", line 330, in _process_response\n    return APIResponse(\n\n\n  File \"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py\", line 145, in parse\n    raise ValueError(exc) from None  # validation error here\n\n\nValueError: 'configuration'"
Traceback (most recent call last):
  File "/Users/ming/Documents/work/repo/tmp/tmp/llm_sample/langChain/localLLM/testOpenLLM.py", line 58, in <module>
    generated = llm_chain.run(product="mechanical keyboard")
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/base.py", line 512, in run
    return self(kwargs, callbacks=callbacks, tags=tags, metadata=metadata)[
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/base.py", line 312, in __call__
    raise e
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/base.py", line 306, in __call__
    self._call(inputs, run_manager=run_manager)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/llm.py", line 103, in _call
    response = self.generate([inputs], run_manager=run_manager)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain/chains/llm.py", line 115, in generate
    return self.llm.generate_prompt(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 516, in generate_prompt
    return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 635, in generate
    params = self.dict()
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_core/language_models/llms.py", line 990, in dict
    starter_dict = dict(self._identifying_params)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/langchain_community/llms/openllm.py", line 222, in _identifying_params
    self.llm_kwargs.update(self._client._config())
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_http.py", line 66, in _config
    self.__config = self._metadata.configuration
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_http.py", line 60, in _metadata
    self.__metadata = self._post(path, response_cls=Metadata, json={}, options={'max_retries': self._max_retries})
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py", line 445, in _post
    return self.request(response_cls, RequestOptions(method='POST', url=path, json=json, **options), stream=stream, stream_cls=stream_cls)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py", line 369, in request
    return self._request(response_cls=response_cls, options=options, remaining_retries=remaining_retries, stream=stream, stream_cls=stream_cls)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py", line 401, in _request
    return self._process_response(response_cls=response_cls, options=options, raw_response=response, stream=stream, stream_cls=stream_cls)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py", line 330, in _process_response
    return APIResponse(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/openllm_client/_shim.py", line 145, in parse
    raise ValueError(exc) from None  # validation error here
ValueError: 'configuration'


'''


#print(llm)

