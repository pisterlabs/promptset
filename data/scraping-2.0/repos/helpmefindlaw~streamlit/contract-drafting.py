# Contract Drafting App
# ---------------------
# This app uses the HelpMeFindLaw API to build an AutoGPT style agent the 
# completes legal research prior to attempting to draft any clauses for
# a end user.

# --------- utils/api.py ---------

import httpx
from pydantic import BaseModel

class HelpMeFindLawCompletionInput(BaseModel):
    prompt: str

class HelpMeFindLawRetrievalInput(BaseModel):
    prompt: str

class HelpMeFindLawClient(BaseModel):
    token: str
    base_url: str = "https://api.helpmefindlaw.com"

    def _run(self, endpoint: str, body) -> str:
        """Use the tool."""
        with httpx.Client(timeout=None) as client:
            resp = client.post(f"{self.base_url}/{endpoint}",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.token}"
                }
            )
            if not resp.status_code == 200:
                raise Exception(resp.text)
            return resp.json()
    
    async def _arun(self, endpoint: str, query: str) -> str:
        """Use the tool asynchronously."""
        body = query
        async with httpx.AsyncClinet(timeout=None) as client:
            resp = client.post(f"{self.base_url}/{endpoint}",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.token}"
                }
            )
            if not resp.status_code == 200:
                raise Exception(resp.text)
            return resp.json()
        
    def completion(self, body: HelpMeFindLawCompletionInput) -> str:
        return self._run("completion", body.model_dump_json())
    
    async def acompletion(self, body: HelpMeFindLawCompletionInput) -> str:
        return await self._arun("completion", body.model_dump_json())
    
    def retrieve(self, body: HelpMeFindLawRetrievalInput) -> str:
        return self._run("retrieve", body.model_dump_json())
    
    async def aretrieve(self, body: HelpMeFindLawRetrievalInput) -> str:
        return await self._arun("retrieve", body.model_dump_json())
    
# --------- utils/tools.py ---------

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel
from typing import Optional, Type

class HelpMeFindlLawCompletionTool(BaseTool):
    name = "helpmefindlaw"
    description = "useful tool for researching the law and are happy with summarised outputs"
    args_schema: Type[BaseModel] = HelpMeFindLawCompletionInput
    client: HelpMeFindLawClient

    def _run(
        self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.client.completion(HelpMeFindLawCompletionInput(prompt=prompt))

    async def _arun(
        self, prompt: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return await self.client.acompletion(HelpMeFindLawCompletionInput(prompt=prompt))
    
    
class HelpMeFindlLawRetreivalTool(BaseTool):
    name = "helpmefindlaw"
    description = "useful tool for researching the law and doing your own analysis over raw text"
    args_schema: Type[BaseModel] = HelpMeFindLawCompletionInput
    client: HelpMeFindLawClient

    def _run(
        self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.client.retrieve(HelpMeFindLawRetrievalInput(prompt=prompt))

    async def _arun(
        self, prompt: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return await self.client.aretrieve(HelpMeFindLawRetrievalInput(prompt=prompt))

# --------- app ---------

from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st

def handle_output(output):
    if isinstance(output, str):
        return output
    elif isinstance(output, dict):
        if output.get("completion"):
            return output["completion"]
    return output

st.set_page_config(page_title="HelpMeFindLaw\n\n BabyAGI For Contract Drafting", page_icon="🦜")
st.title("HelpMeFindLaw x Langchain 🦜")
st.subheader("AutoGPT For Contract Drafting")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
hmfl_api_key = st.sidebar.text_input("HelpMeFindLAw API Key", type="password")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step)
                # st.write(step[0].log)
                # if step[1].get("completion"):
                #     st.write(step[1]["completion"])
                # else:
                #     st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Draft a non-compete clause for an employment contract in Texas"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not hmfl_api_key:
        st.info("Please add your HelpMeFindLaw API key to continue.")
        st.stop()

    client = HelpMeFindLawClient(token=hmfl_api_key)
    model = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
    tools = [
        DuckDuckGoSearchRun(name="Search"),
        HelpMeFindlLawCompletionTool(client=client)
    ]
    planner = load_chat_planner(llm=model)
    executor = load_agent_executor(llm=model, tools=tools)
    agent = PlanAndExecute(planner=planner, executor=executor)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = agent({ "input": prompt }, callbacks=[st_cb])

        output = response["output"]
        if isinstance(output, str):
            st.write(output)
        else:
            st.write(output["action_input"])
        if response.get("intermediate_steps"):
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]