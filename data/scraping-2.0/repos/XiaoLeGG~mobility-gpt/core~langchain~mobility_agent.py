from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import MessagesPlaceholder
from core.langchain.callback_manager import MACallbackHandler
from core.langchain.tool_manager import collect_tools
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import uuid
import os

class MobilityAgent():

    def __init__(self):
        self._mcbh = MACallbackHandler(self)
        self._cb = CallbackManager([StdOutCallbackHandler(), self._mcbh])
        self._llm = ChatOpenAI(temperature=0, model="gpt-4")
        self._tools = collect_tools()

        # self._json_template = """
        # {
        #     "step": 1,
        #     "tool": "tool1",
        #     "parameters": {
        #         "args1": "value1",
        #         "args2": "value2"
        #     },
        #     "thought": "reason",
        # },
        # {
        #     "step": 2,
        #     "tool": "tool2",
        #     "parameters": {
        #         "args1": "value1",
        #         "args2": "value2"
        #     },
        #     "thought": "reason",
        # },
        # """

        # self._prompt = """Here are some information you need to know before our work:
        # [ROLE] You are a spatio-temporal data analyst. You are knowledged about the mobility data analysing. You are now chatting with a user.
        # [REQUEST TYPES]
        # 1. [DATA PROCESSING, ANALYSING AND VISUALIZATION] You need to process the data with tools and give feedback to the user.
        # 2. [KNOWLEDGE QUERY] You need to answer the question with knowledge and give feedback to the user.
        # [HANDLING STEPS WHEN NEW REQUEST COMES]
        # 1. Read the request.
        # 2. Analyse the situation, problem and attention point of the request.
        # 3. Determine the request type.
        # 4. Handle the request based on different request types.
        # 5. For ambiguous request, you can ask for more information before you handle the request.
        # [OUTPUT DATA FILE] Every output file should be csv file.
        # [OUTPUT FOLDER] "output/"
        # [INITIAL DATA FILE] {input_file}
        # [TIPS YOU SHOULD KNOW]
        # 1. Preprocess the data when you think it is necessary.
        # 2. Write your personal codes with PythonREPLTool when there is no tool able to solve the request but remember to run TableReaderTool to read the information of the table file to help you better write codes.
        # 3. For PythonREPLTool, you should store the processed data in a csv file for the next step to use.
        # 4. Remember to consider the data features memtioned in the request when choosing parameters.
        # 5. The seperate symbols of the data file is ",".
        # [MISTAKES YOU SHOULD AVOID]
        # 1. Overestimate the function of tools. Each tool should have been considered its parameters for better estimating its function when you choose it.
        # 2. Change the column name of latitude, longitude, datetime and uid. You should keep the table column names.
        # 3. Change the data format of table data file. You should keep it as a well formated table csv file.
        # 4. Output a csv file without header. You should keep the header of the output csv file.

        # Begin!
        # """

        self._prompt = """Before we begin, here are some key points:
        [ROLE] You're a spatio-temporal data analyst specializing in mobility data. You are now chatting with user.
        [REQUEST TYPES]
        1. [DATA PROCESSING, ANALYSING AND VISUALIZATION] Process, analyze, and visualize data.
        2. [KNOWLEDGE QUERY] Answer knowledge-based questions.
        [HANDLING STEPS FOR NEW REQUESTS]
        1. Understand the request.
        2. Analyze the problem and its context.
        3. Identify the request type.
        4. Handle the request accordingly.
        5. For unclear requests, seek more information.
        [OUTPUT] All output files should be CSV, stored in "{output_folder}".
        [INITIAL DATA FILE] {input_file}
        [TIPS]
        1. Preprocess data if necessary.
        2. Use MobilityGPT_Python_REPL for custom code, but use table_reader first to read table data.
        3. When using MobilityGPT_Python_REPL, you should outputs a CSV file in code.
        4. Consider data features when choosing parameters.
        5. Data file separators are ','.
        6. When processing and saving data in MobilityGPT_Python_REPL, use Pandas DataFrame instead of Series.
        Examples for tip 6:
        - Series type data:
        ```
        # output_file = your output path
        first_record = data.iloc[0]
        first_record.to_frame().to_csv(output_file, index=False)
        ```
        - DataFrame type data:
        ```
        # output_file = your output path
        data.to_csv(output_file, index=False)
        ```
        [COMMON MISTAKES]
        1. Don't overestimate tool capabilities. Consider parameters for optimal use.
        2. Don't change column names for latitude, longitude, datetime, and uid.
        3. Don't alter the data format. Keep it as a well-formatted CSV table.
        4. Don't output CSV files without headers.
        
        Begin!"""

        # self._suffix="""

        # [CHAT_HISTORY]
        # {memory}
        # [REQUEST] {request}
        # {agent_scratchpad}
        # """

        # self._zero_shot_prompt = ZeroShotAgent.create_prompt(
        #     tools=self._tools,
        #     prefix=self._prompt,
        #     suffix=self._suffix,
        #     input_variables=["input_file", "request", "memory", "agent_scratchpad"],
        # )

        self._agent_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        # self._llm_chain = LLMChain(llm=self._llm, prompt=self._zero_shot_prompt)
        # self._agent = ZeroShotAgent(
        #     llm_chain=self._llm_chain,
        #     tools=self._tools,
        #     verbose=True
        # )
        self._agent = initialize_agent(
            tools=self._tools,
            llm=self._llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            callback_manager=self._cb,
            agent_kwargs = {
                'extra_prompt_messages': [MessagesPlaceholder(variable_name="memory")]
            },
            memory=self._agent_memory,
            handle_parsing_errors=lambda e: "Error occurs, you may check the existance of file and use table reader to check the data: " + str(e),
        )

        self._is_started = False
    
    def start(self, input_file: str) -> uuid.UUID:
        """Start the agent.
        
        Parameters
        ----------
        input_file : str
            The initial data file path to be processed.
        
        """
        self._current_session_id = uuid.uuid4()
        self._agent_memory.clear()
        self._conversation_count = 0
        self._output_folder = 'output/' + str(self._current_session_id) + '/'
        os.makedirs(self._output_folder)
        self._is_started = True
        response = self._agent.run(self._prompt.format(input_file=input_file, output_folder=self._output_folder))
        return response, self._current_session_id

    def ask(self, request: str):
        """Ask the agent to solve the problem.

        Parameters
        ----------
        request : str
            The request of the user.

        Returns
        -------
        str
            The output file path.
        """
        if not self._is_started:
            raise Exception("Please start the agent first.")
        self._conversation_count += 1
        response = self._agent.run(request)
        return response