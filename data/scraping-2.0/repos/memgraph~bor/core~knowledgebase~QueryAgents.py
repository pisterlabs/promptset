from __future__ import annotations

from typing import List

import os
from pathlib import Path

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

from core.knowledgebase import constants
from core.knowledgebase.MemgraphManager import MemgraphManager
from core.knowledgebase.notes.Searcher import Searcher


class GeneralQueryAgent:
    def __init__(self: GeneralQueryAgent, repo_path: str, tools: List[Tool]) -> None:
    
        self.repo_path = repo_path

        self.system_message = ''
        self._init_system_message()

        self.llm = ChatOpenAI(
            temperature=constants.LLM_MODEL_TEMPERATURE, 
            openai_api_key=constants.OPENAI_API_KEY,
            model_name=constants.LLM_MODEL_NAME
        )
        
        self.agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": self.system_message, 
        }

        self.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        self.run_cypher_query = Tool.from_function(
                func = MemgraphManager.select_query_tool,
                name = "run_cypher_query",
                description = f"""Useful when you want to run Cypher queries on the knowledge graph. 
                Note that the input has to be valid Cypher. Consult the graph schema in order to know how to write correct queries. 
                Pay attention to the repo_path attribute.
                Returns results of executing the query."""
        )

        self.tools = tools + [self.run_cypher_query]
        self._init_agent(self.tools)

        return

    def _init_agent(self: GeneralQueryAgent, tools: List[Tool]) -> None:
        self.agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=self.agent_kwargs,
            memory=self.memory
        )

        return

    def _init_system_message(self: GeneralQueryAgent) -> None:
            prompt_name = 'system_message_query'
            prompt_path = Path(os.path.join(os.path.dirname(__file__), 'prompts', prompt_name))
            prompt_text = prompt_path.read_text()
            prompt_template = PromptTemplate.from_template(prompt_text)

            mm = MemgraphManager()
            schema = mm.get_schema_for_repo(self.repo_path)
            self.system_message = SystemMessage(content=prompt_template.format(schema=schema, repo_path=self.repo_path))

            return

    def ask(self: GeneralQueryAgent, question: str) -> str:
        return self.agent.run(question)


class NotesQueryAgent(GeneralQueryAgent):

    def __init__(self: NotesQueryAgent, repo_path: str) -> None:

        self.searcher = Searcher(repo_path)

        search_graph = Tool.from_function(
            func=self.searcher.search_graph_tool,
            name="search_graph",
            description="""Useful for when you want to query the knowledge graph via the query embeddings. 
            Returns descriptions of 3 most similar nodes."""
        )


        search_text = Tool.from_function(
                func=self.searcher.search_text_tool,
                name="search_text",
                description="""Useful for when you want to query the document store via the query embeddings. 
                Returns 3 most similar sentences to a given query."""
        )

        notes_tools = [search_graph, search_text]
        super().__init__(repo_path, notes_tools)

        return


class CodeQueryAgent(GeneralQueryAgent):

    def __init__(self: CodeQueryAgent, repo_path: str) -> None:
        
        read_file = Tool.from_function(
            func=lambda p: Path(p).read_text(),
            name="read_file",
            description="""Useful for when you want to read the entire contents of a text file. 
            Provide the tool with the absolute path of the file you want to read.""",
        )

        listdir = Tool.from_function(
            func = CodeQueryAgent.list_files,
            name="listdir", 
            description="""Useful for when you want to find out the directory structure of the repository, and to know where a file is located."""
        )
        
        code_tools = [read_file, listdir]
        super().__init__(repo_path, code_tools)
        
        return

    @staticmethod
    def list_files(startpath: str) -> str:
        out = ""
        for root, dirs, files in os.walk(startpath):
            if '.git' in dirs:
                dirs.remove('.git')
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            out += '{}{}/\n'.format(indent, os.path.basename(root))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                out += '{}{}\n'.format(subindent, f)
        return out


if __name__ == '__main__':

    example_reponame = 'History'
    example_repopath = os.path.join(os.path.dirname(__file__), 'examples', example_reponame)
    na = NotesQueryAgent(example_repopath)


    print(na.ask("hi"))
    print(na.ask("I am Patrik."))
    print(na.ask("What did I tell you my name was?"))

    print(na.ask("When was Napoleon born?"))
    print(na.ask("Where was Napoleon born?"))
    print(na.ask("Can you quote the sentence available in the documents with tells about the birth of Napoleon?"))
