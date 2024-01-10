from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from typing import Any, Dict, List


Traceloop.init(app_name="langchain-obs", disable_batch=True)

'''
comet_callback = CometCallbackHandler(
    project_name="example-langchain",
    complexity_metrics=True,
    stream_logs=True,
    tags=["llm"],
    visualizations=["dep"],
)
'''

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        

myCallback = MyCustomHandler()

callbacks = [StdOutCallbackHandler(), myCallback]
llm = OpenAI(temperature=0.9, callbacks=callbacks, verbose=True)

template = "Say my name: {name}"
prompt_template = PromptTemplate(input_variables=["name"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, callbacks=callbacks)

test_prompts = [{"name": "Wolfgang"}]
print(synopsis_chain.apply(test_prompts))


