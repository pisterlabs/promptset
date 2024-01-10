import streamlit as st
import wikipedia
import faiss
import numpy as np
import pandas as pd
import base64
import json
import os
import tempfile
import pinecone 
import requests
import re
import time
from pydantic import BaseModel, Field
from collections import deque
from typing import List, Union,Callable,Dict, Optional, Any
from langchain.agents import  ZeroShotAgent,AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent, Tool,AgentType,create_pandas_dataframe_agent
from langchain.prompts import StringPromptTemplate,PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish,Document
from langchain.document_loaders import PyPDFLoader
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import OpenAI,BaseLLM
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.vectorstores.base import VectorStore
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper,WikipediaAPIWrapper,TextRequestsWrapper
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

st.set_page_config(
    page_title="可读-财报GPT",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "可读-财报GPT"
    }
)
显示 = ""
class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create the main task with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
            " Be careful,This model's maximum context length is 3900 tokens."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
class TaskPrioritizationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " 1"
            " 2"
            " Start the task list with number {next_task_id}."
            " Be careful,This model's maximum context length is 3800 tokens."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
def get_next_task(task_creation_chain: LLMChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task']) for item in sorted_results]

def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)
class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    def add_task(self, task: Dict):
        self.task_list.append(task)
    def print_task_list(self):
        print("\n*****TASK LIST*****\n")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])
    def print_next_task(self, task: Dict):
        print("\n*****NEXT TASK*****\n")
        print(str(task["task_id"]) + ": " + task["task_name"])
    def print_task_result(self, result: str):
        global 显示
        print("\n*****TASK RESULT*****\n")
        print(result)
        显示+=result
    @property
    def input_keys(self) -> List[str]:
        return ["objective"]
    @property
    def output_keys(self) -> List[str]:
        return []
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()
                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)
                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)
                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )
                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\n*****TASK ENDING*****\n")
                break
        return {}
    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )
wikipedia.set_lang("zh")
template3 = """尽量以少的token准确快速给出Final Answer,You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do,尽量少的token
    Action: the action to take, should be one of [{tool_names}],尽量少的token
    Action Input: the input to the action,尽量少的token
    Observation: the result of the action,尽量少的token
    Thought: I now know the final answer,尽量少的token
    Final Answer: the final answer to the original input question,准确快速给出Final Answer
    All inputs 、output and context tokens are totally limited to 3800.
    Question: {input}
    {agent_scratchpad}
    都用中文表示，除了格式中的Question:Thought:Action:Action Input:Observation:Thought:Final Answer
    """
with st.sidebar:
    st.info('请先输入正确的openai api-key')
    st.text_input('api-key','', key="input_api")
    with st.expander("ChatOpenAI属性设置"):
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3)
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="在OpenAI GPT语言模型中，温度（temperature）是一个用于控制生成文本随机性和多样性的参数。它可以被理解为对下一个词的概率分布进行重新加权的因子，其中较高的温度值将导致更多的随机性和不确定性，而较低的温度值将导致更少的随机性和更高的确定性。通过调整温度值，可以控制生成文本的风格和多样性，以满足特定的应用需求。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3)
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.3)
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0)
st.title('智能财报（中国上市公司）')
tab1, tab2 = st.tabs(["问答模式(QA)", "任务模式（BabyAGI）"])
wikipedia = WikipediaAPIWrapper()
search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCLKh_M6oShQ6rUJiw8UeQ74M39tlCUa9M",google_cse_id="c147e3f22fbdb4316")
logo_url = "https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png"
if st.session_state.input_api:
    llm=ChatOpenAI(
        model_name=model,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        openai_api_key=st.session_state.input_api
    )
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.input_api)
    def 中国平安年报查询(input_text):
        pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                          environment="us-east4-gcp",  # next to api key in console
                          namespace="ZGPA_601318")
        index = pinecone.Index(index_name="kedu")
        a=embeddings.embed_query(input_text)
        www=index.query(vector=a, top_k=1, namespace='ZGPA_601318', include_metadata=True)
        c = [x["metadata"]["text"] for x in www["matches"]]
        return c
    def 双汇发展年报查询(input_text):
        namespace="ShHFZ_000895"
        pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                          environment="us-east4-gcp",  # next to api key in console
                          namespace=namespace)
        index = pinecone.Index(index_name="kedu")
        a=embeddings.embed_query(input_text)
        www=index.query(vector=a, top_k=1, namespace=namespace, include_metadata=True)
        c = [x["metadata"]["text"] for x in www["matches"]]
        return c  
    wiki_tool = Tool(
                name="维基",
                func=wikipedia.run,
                description="当您需要搜索百科全书时，这个工具非常有用。输入转换为英文，输出转换为中文"
                 )
    search_tool =  Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要搜索互联网时，这个工具非常有用。"
                )
    zgpa_tool =  Tool(
                    name = "ZGPA",
                    func=中国平安年报查询,
                    description="当您需要回答有关中国平安(601318)中文问题时，这个工具非常有用。"
                )
    shhfz_tool =  Tool(
                    name = "ShHFZ",
                    func=双汇发展年报查询,
                    description="当您需要回答有关双汇发展(000895)中文问题时，这个工具非常有用。"
                )
    ALL_TOOLS = [zgpa_tool,shhfz_tool,search_tool,wiki_tool]
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    def get_tools(query):
        docs = retriever.get_relevant_documents(query)
        return [ALL_TOOLS[d.metadata["index"]] for d in docs]
    global qa
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer in Chinese:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools_getter: Callable
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            return self.template.format(**kwargs)
    class CustomPromptTemplate_Upload(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)
    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    output_parser = CustomOutputParser()
    @st.cache_resource
    def 分析(input_text):
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file.flush()
                loader = PyPDFLoader(tmp_file.name)
        elif input_text != "":        
            loader = PyPDFLoader(input_text)
        else:
            return None
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    with tab1:
        with get_openai_callback() as cb:
            with st.expander("[可选]上传"):
                file = st.file_uploader("PDF文件", type="pdf")
                input_text = st.text_input('PDF网址', '')
                qa = 分析(input_text)
            input_text1 = st.text_input(':blue[查询]','')
            if st.button('确认'):
                start_time = time.time()
                if not qa:
                    query = input_text1
                    prompt3 = CustomPromptTemplate(
                        template=template3,
                        tools_getter=get_tools,
                        input_variables=["input", "intermediate_steps"])
                    llm_chain = LLMChain(llm=llm, prompt=prompt3)
                    tools = [
                        Tool(
                            name = "ZGPA",
                            func=中国平安年报查询,
                            description="当您需要回答有关中国平安(601318)中文问题时，这个工具非常有用。输入是中文"
                        ),
                        Tool(
                            name = "Google",
                            func=search.run,
                            description="当您需要搜索互联网时，这个工具非常有用。"
                        ),
                        Tool(
                        name="维基",
                        func=wikipedia.run,
                        description="当您需要搜索百科全书时，这个工具非常有用。"
                    ),
                        Tool(
                        name = "ShHFZ",
                        func=双汇发展年报查询,
                        description="当您需要回答有关双汇发展(000895)中文问题时，这个工具非常有用。输入是中文"
                    )]
                    tool_names = [tool.name for tool in tools]
                    agent3 = LLMSingleActionAgent(
                            llm_chain=llm_chain, 
                            output_parser=output_parser,
                            stop=["\nObservation:"], 
                            allowed_tools=tool_names
                        )
                    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True,return_intermediate_steps=True)
                    response = agent_executor({"input":query})
                    st.caption(response["output"])
                    with st.expander("查看过程"):
                        st.write(response["intermediate_steps"])
                else:
                    query = input_text1
                    tools = [Tool(
                        name = "上传",
                        func=qa.run,
                        description="当您需要回答有关上传信息的问题时，这个工具非常有用。"
                        ),
                              Tool(
                        name="维基",
                        func=wikipedia.run,
                        description="当您需要搜索百科全书时，这个工具非常有用。"
                        ),
                              Tool(
                            name = "Google",
                            func=search.run,
                            description="当您需要搜索互联网时，这个工具非常有用。"
                        )
                       ]
                    tool_names = [tool.name for tool in tools]
                    prompt_Upload = CustomPromptTemplate_Upload(
                    template=template3,
                    tools=tools,
                    input_variables=["input", "intermediate_steps"])
                    llm_chain = LLMChain(llm=llm, prompt=prompt_Upload)
                    agent3 = LLMSingleActionAgent(
                        llm_chain=llm_chain, 
                        output_parser=output_parser,
                        stop=["\nObservation:"], 
                        allowed_tools=tool_names
                    )
                    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True,return_intermediate_steps=True)
                    response = agent_executor({"input":query})
                    st.caption(response["output"])
                    with st.expander("查看过程"):
                        st.write(response["intermediate_steps"])
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"项目完成所需时间: {elapsed_time:.2f} 秒")  
                st.write(f"Total Tokens: {cb.total_tokens}") 
                st.write(f"Prompt Tokens: {cb.prompt_tokens}") 
                st.write(f"Completion Tokens: {cb.completion_tokens}") 
                st.write(f"Total Cost (USD): ${cb.total_cost}") 
    with tab2:
        with get_openai_callback() as cb:
            OBJECTIVE = st.text_input('提问','', key="name_input1_2")
            todo_prompt = PromptTemplate.from_template("尽量以少的token准确快速给出这个目标最重要的待办事项： {objective}.")
            todo_chain = LLMChain(llm=OpenAI(temperature=temperature,openai_api_key=st.session_state.input_api), prompt=todo_prompt)
            tools = [
                   Tool(
                                    name = "ZGPA",
                                    func=中国平安年报查询,
                                    description="当您需要回答有关中国平安(601318)中文问题时，这个工具非常有用。输入是中文"
                                ),
                                Tool(
                                    name = "Google",
                                    func=search.run,
                                    description="当您需要搜索互联网时，这个工具非常有用。"
                                ),
                                Tool(
                                name="维基",
                                func=wikipedia.run,
                                description="当您需要搜索百科全书时，这个工具非常有用。"
                            ),
                                Tool(
                                name = "ShHFZ",
                                func=双汇发展年报查询,
                                description="当您需要回答有关双汇发展(000895)中文问题时，这个工具非常有用。输入是中文"
                            ),
                Tool(
                    name = "TODO",
                    func=todo_chain.run,
                    description="此功能可用于创建待办事项清单。输入：要为其创建待办事项清单的目标。输出：该目标的最重要事项的待办事项。请非常清楚地说明目标是什么。!"
                )
            ]
            prefix = """准确快速给出任务的解答: {objective}. 考虑到先前完成的这些任务：{context}."""
            suffix = """Question: {task}
            {agent_scratchpad}
            都用中文表示，除了格式中的提取前缀
            All inputs、output and context tokens are in total limited to 3800.
            最终结果用中文表示.
            """
            prompt = ZeroShotAgent.create_prompt(
                tools, 
                prefix=prefix, 
                suffix=suffix, 
                input_variables=["objective", "task", "context","agent_scratchpad"]
            )
            verbose=True
            # If None, will keep on going forever
            max_iterations: Optional[int] = 3
            index = faiss.IndexFlatL2(1536)
            vectorstore = FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})
            baby_agi = BabyAGI.from_llm(
                llm=llm,
                vectorstore=vectorstore,
                verbose=verbose,
                max_iterations=max_iterations
            )
            if st.button('问答'):
                start_time = time.time()
                baby_agi({"objective": OBJECTIVE})
                text_list = 显示.split('\n')
                for i in text_list:
                    st.write(i)
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"项目完成所需时间: {elapsed_time:.2f} 秒")  
                st.write(f"Total Tokens: {cb.total_tokens}") 
                st.write(f"Prompt Tokens: {cb.prompt_tokens}") 
                st.write(f"Completion Tokens: {cb.completion_tokens}") 
                st.write(f"Total Cost (USD): ${cb.total_cost}") 
