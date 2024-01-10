"""
Using langchains create a task to check the security
of a login form found at a specific web address

THIS IS DANGEROUS TO RUN

Uses the hydra cli program
https://www.cyberpunk.rs/password-cracker-thc-hydra
"""
import os
import uuid
import sys
import random
from urllib.parse import urlparse
from datetime import datetime
import os
import logging
import sys

from langchain.agents import Tool
from langchain.utilities import BashProcess
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
# from langchain.tools import ShellTool

# from langchain.vectorstores import FAISS
# import faiss
# from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.redis import Redis
import redis

from langchain.embeddings import OpenAIEmbeddings

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

from tools.stream_to_logger import StreamToLogger

class LoginChecker:
    def __init__(self, http_url):
        self.uuid = str(uuid.uuid4()).replace('-', '')

        self.autogpt_resp = ";_; failed"

        # prompt for the agent to use, will be a list
        data_path = os.path.abspath("tools/data/")
        logs_path = os.path.abspath("tools/logs/")
        # bin_path = os.path.abspath("tools/bin")

        self.http_url = http_url

        # setup logging
        self.logging_file_name = f"lc_runlog{datetime.now().strftime('%Y%m%d_%H%M')}_{self.uuid}.txt"
        self.logging_file_path =  f"{logs_path}/{self.logging_file_name}"
        self.summary_file_name = f"lc_security_report{datetime.now().strftime('%Y%m%d_%H%M')}_{self.uuid}.txt"
        self.summary_file_path = f"{logs_path}/{self.summary_file_name}"
        
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if not os.path.exists(self.logging_file_path):
            open(self.logging_file_path, "w").close()
        
        logging.basicConfig(
            filename=self.logging_file_path,
            level=logging.INFO,
            format="\n%(message)s\n")

        self.logging = logging.getLogger(__name__)
        
        self.tools = [
            Tool(
                "search",
                GoogleSearchAPIWrapper().run,
                """
                Useful for when you need to answer questions about current events. 
                You should ask targeted questions
                """
            ),
            # PythonREPLTool(),
            # ShellTool(),
            Tool(
                "bash",
                BashProcess().run,
                "useful for when you want to run a command in the bash terminal."
            ),
            WriteFileTool(),
            ReadFileTool(),
            Tool(
                "search2",
                DuckDuckGoSearchRun().run,
                """
                Useful for when you need to answer questions about current events. 
                You should ask targeted questions
                Use this when search hits a quota limit
                """

            )
        ]
        
        self.embeddings = OpenAIEmbeddings()

        self.error_log_path = f"{logs_path}/lc_error{datetime.now().strftime('%Y%m%d_%H%M')}_{self.uuid}.txt"
        if not os.path.exists(self.error_log_path):
            open(self.error_log_path, 'w').close()

        self.info_log_path =  f"{logs_path}/lc_info{datetime.now().strftime('%Y%m%d_%H%M')}_{self.uuid}.txt"
        if not os.path.exists(self.info_log_path):
            open(self.info_log_path, "w").close()

        urlparts = urlparse(self.http_url)
        hostport = urlparts.netloc.split(":")
        if len(hostport) == 2:
            hydra_host = f"-s {hostport[1]} {hostport[0]}"
        else:
            hydra_host = f"{hostport[0]}"

        self.goals = [
            # f"Stay on task with your goals and don't get into a loop. If you are looping, break the loop and re-exam your goals that you have left",
            # f"Check {self.http_url} if it is a government, corporate or social network website by searching the web. Use search2 if search is hitting a quota limit. If the URL contains .gov or .org then it is government or corporate. If it is, stop all tasks and finish with reply \"Not legal to run prompt check on {self.http_url}. Its a government or corporate website!!\"",
            # f"Do a search on cyber security laws and understand what is legal or not to hack and not get in trouble. Use search2 if search is hitting a quota limit.",
            # f"Do a search on what {self.http_url} is about. Use search2 if search is hitting a quota limit. If there is no information about it then continue on to the next step. If it is found to get you in trouble, stop all tasks and finish with a reply to the user \"Not legal to run prompt check on {self.http_url}!!\".",
            # f"Check if {self.http_url} is a valid url to a login form and if not, stop all tasks and finish with a reply to user \"Not a valid form URL. Please provide a actual form URL\"",
            # f"Check if log files {self.info_log_path} and {self.error_log_path} exist and if not, create them",
            f"""
            Run the command below. Don't try to install hydra. If hydra command failed, move on to step 2. Do not use sudo. Be very careful about running this.
            ```bash 
            hydra -v -L {data_path + "/username_list_small.txt"} -P {data_path + "/password_list_small.txt"} {hydra_host} http-post-form "/admin/login/:username=^USER^&password=^PASS^:F=Invalid username or password" > {self.info_log_path} 2> {self.error_log_path}
            ```
            """,
            # f"""
            # If step 1 failed, try this step. If not, continue to step 3. Look into using the selenium python library via REPL. Use the "write_file" command. Using the source of {self.http_url}, write a python program using selenium python and its WebDriver API at {data_path} with name login_test.py. The program has to iterate through the username list {data_path + "/username_list_small.txt"} with every password at {data_path + "/password_list_small.txt"} and try to login at {self.http_url}. Store the python program at {data_path}. Use bash symbols > and 2> to stream to the stdout log {self.info_log_path} and the stderr log {self.error_log_path}. If this doesn't work continue on to next step.
            # """,
            f"""
            If any of the above steps worked, write a summary security report locally at
                {self.summary_file_path}
            using the information log 
                {self.info_log_path} 
            
            Include suggestions, if needed, on what can be done to fix issues.
            
            If the information log is empty just write in report 
                "No security issues found with form at {self.http_url}"

            If the previous steps failed, just write in report
                "FAILURE - No security tools found on machine"
            
            Avoid using any interactive editors

            Finish after report is written
            """,
            # "Congrats, you have completed all the tasks successfully, once the report is created, stop all other tasks"
        ]

        try:
            sys.stdout = StreamToLogger(self.logging, logging.INFO)
            # check if index name exists and if not create it
            # connect to Redis server
            # redis_check = redis.Redis.from_url(os.environ["REDIS_URL"])

            # check if the index exists
            # if len(redis_check.keys(
            #     "doc:{}*".format(redis_idx_name)
            # )) == 0:
            # create the index if it doesn't exist

            Redis.from_texts(
                texts=["hacker"],
                redis_url=os.environ["REDIS_URL"],
                index_name=self.uuid,
                embedding=self.embeddings
            )

            self.vectorstore = Redis(
                redis_url=os.environ["REDIS_URL"],
                index_name=self.uuid,
                embedding_function=self.embeddings.embed_query
            )

            # using faiss
            # possibly can use redis but will need to update
            # the landchain agent.py in experimental for autogpt
            # to use add_text
            # embedding_size = 1536
            # index = faiss.IndexFlatL2(embedding_size)
            # self.vectorstore = FAISS(self.embeddings.embed_query, index, InMemoryDocstore({}), {})


        except Exception as err:
            print("Redis creation failed {err}")
            # print("FAISS creation failed {err}")
            # yield err
            raise err
        
    
    def run(self):
        llm = ChatOpenAI(temperature=0, streaming=True)

        agent = AutoGPT.from_llm_and_tools(
            ai_name=self.uuid,
            ai_role="Penetration Tester",
            tools=self.tools,
            llm=llm,
            memory=self.vectorstore.as_retriever()
        )
        agent.chain.verbose = False

        try:
            self.autogpt_resp = agent.run(self.goals)
        except Exception as err:
            print(f"AutoGPT failure {err}")

        if self.autogpt_resp:
            self.logging.info(f"AutoGPT Response: {self.autogpt_resp}")
