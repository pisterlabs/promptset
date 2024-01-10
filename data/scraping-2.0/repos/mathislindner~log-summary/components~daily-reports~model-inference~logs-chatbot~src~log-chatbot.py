#use a langchain agent to trasverse the df of the logs to answer questions
# for example: what happened to the host "" in the last 15 minutes?
# retrieves the logs of the host in the last 15 minutes
# summarize the logs
# return the summary...

from typing import Any
import requests
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd
#agents and prompts
from langchain.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
#llm class
from langchain.llms.base import LLM
from langchain.schema.messages import  BaseMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

class flaskllm(LLM):
    #super().__init__()
    ip : str = "localhost"
    port : int = 80
    suburl : str = "query"
    endpoint_url :str = "http://{}:{}/{}".format(ip, port, suburl)
    headers : str = {'Content-type': 'application/json'}
    
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        data = {"userQuery": messages}
        r = requests.post(self.endpoint_url, json=data, headers=self.headers, timeout=360)
        print(r.json())
        return r.json() #[0]["generated_text"]
    
    @property
    def _llm_type(self) -> str:
        return "flask-bot"

llm = flaskllm(ip ="localhost", port = 80)

tools = load_tools(['python_repl_ast'])


df_errors= pd.read_csv("/data/preprocessed/logs/2023-06-28/error.csv")
df_warnings= pd.read_csv("/data/preprocessed/logs/2023-06-28/warning.csv")


agent = create_pandas_dataframe_agent(llm, [df_errors, df_warnings])
print(agent.run("get me one random error", verbose = True))
