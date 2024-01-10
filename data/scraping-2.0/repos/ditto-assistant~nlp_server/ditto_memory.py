from datetime import datetime
import os
import pickle
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import logging


from dotenv import load_dotenv

load_dotenv()

# import google search agent
from google_search_agent import GoogleSearchAgent

# import example store
from ditto_example_store import DittoExampleStore

# import short term memory store
from ditto_stmem import ShortTermMemoryStore

# import knowledge graph agent job
from modules.kg_job import KGJob

# import llm programmer agent
from modules.programmer_agent import ProgrammerAgent

# import llm openscad agent
from modules.openscad_agent import OpenSCADAgent

# import llm code compiler
from modules.llm_code_compiler import write_llm_code, run_llm_code, run_openscad_code

log = logging.getLogger("ditto_memory")
logging.basicConfig(level=logging.INFO)

from templates.llm_tools import LLM_TOOLS_TEMPLATE
from templates.default import DEFAULT_TEMPLATE

KG_MODE = bool(os.environ["knowledge_graph"])


class DittoMemory:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.kg_mode = KG_MODE
        self.__handle_params()
        self.short_term_mem_store = ShortTermMemoryStore()
        self.google_search_agent = GoogleSearchAgent(verbose=verbose)
        self.programmer_agent = ProgrammerAgent()
        self.openscad_agent = OpenSCADAgent()
        self.example_store = DittoExampleStore()
        self.memory = {}

    def __handle_params(self):
        self.llm_provider = os.environ["LLM"]
        self.template_type = (
            "DEFAULT" if "SERPER_API_KEY" not in os.environ else "SELF-ASK-WITH-SEARCH"
        )
        self.template = (
            DEFAULT_TEMPLATE
            if "SERPER_API_KEY" not in os.environ
            else LLM_TOOLS_TEMPLATE
        )
        if self.llm_provider == "huggingface":
            # repo_id = "google/flan-t5-xxl"
            repo_id = "codellama/CodeLlama-13b-hf"
            self.llm = HuggingFaceHub(
                repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 3000}
            )
        else:  # default to openai
            self.llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo-16k")

        # check if code compiler is enabled
        self.llm_code_compiler = bool(os.environ["COMPILE_CODE"])
        # set code compiler user
        self.llm_code_compiler_user = os.environ["COMPILE_CODE_USER"]

    def __create_load_memory(self, reset=False, user_id="Human"):
        mem_dir = f"memory/{user_id}"
        mem_file = f"{mem_dir}/ditto_memory.pkl"
        if not os.path.exists(mem_dir):
            os.makedirs(mem_dir)
            log.info(f"Created memory directory for {user_id}")
        if not os.path.exists(mem_file) or reset:
            if self.llm_provider == "openai":
                embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
                index = faiss.IndexFlatL2(embedding_size)
                embedding_fn = OpenAIEmbeddings().embed_query
                vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
            else:
                embedding_size = 768  # Dimensions of the HuggingFaceEmbeddings
                index = faiss.IndexFlatL2(embedding_size)
                embedding_fn = HuggingFaceEmbeddings().embed_query
                vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
            retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))
            self.memory[user_id] = VectorStoreRetrieverMemory(retriever=retriever)
            self.memory[user_id].save_context(
                {f"Human": "Hi! What's up? "},
                {"Ditto": "Hi! My name is Ditto. Nice to meet you!"},
            )
            self.memory[user_id].save_context(
                {f"Human": "Hey Ditto! Nice to meet you too. Glad we can be talking."},
                {"Ditto": "Me too!"},
            )
            self.memory[user_id].save_context(
                {f"Human": "Write me a python script that says hello world."},
                {"Ditto": "<PYTHON_AGENT> write a script that says hello world"},
            )
            self.memory[user_id].save_context(
                {f"Human": "What is the current time in New York?"},
                {"Ditto": "<GOOGLE_SEARCH> current time in New York"},
            )
            self.memory[user_id].save_context(
                {f"Human": "Can you make me a computer mouse in OpenSCAD?"},
                {"Ditto": "<OPENSCAD_AGENT> make me a computer mouse in OpenSCAD."},
            )
            pickle.dump(self.memory[user_id], open(mem_file, "wb"))
            log.info(f"Created memory file for {user_id}")
        else:
            self.memory[user_id] = pickle.load(open(mem_file, "rb"))
            log.info(f"Loaded memory file for {user_id}")

    def save_new_memory(self, prompt, response, user_id="Human", face_name="none"):
        user_name = face_name if not face_name == "none" else user_id
        self.__create_load_memory(user_id=user_id)
        mem_dir = f"memory/{user_id}"
        mem_file = f"{mem_dir}/ditto_memory.pkl"
        self.memory[user_id].save_context({f"{user_name}": prompt}, {"Ditto": response})
        pickle.dump(self.memory[user_id], open(mem_file, "wb"))
        log.info(f"Saved new memory for {user_id}")

    def reset_memory(self, user_id="Human"):
        self.__create_load_memory(reset=True, user_id=user_id)
        log.info(f"Reset memory for {user_id}")

    def add_example_to_query(self, query, stmem_query, examples):
        query_prefix = (
            "\n(You do not need to use these past memories if not relevant)"
            + "\n\nTools:\n"
            + "Ditto has access to the following tools, and they can be used in the following ways:\n"
            + "1. GOOGLE_SEARCH: <GOOGLE_SEARCH> <query>\n"
            + "1.a GOOGLE_SEARCH can be used to search the web for information. Only use this tool if the user's prompt can be better answered by searching the web."
            + "\n"
            + "2. PYTHON_AGENT: <PYTHON_AGENT> <query>\n"
            + "2.a PYTHON_AGENT can be used to program a script for the user. The script will be compiled and run for the user."
            + "3. OPENSCAD_AGENT: <OPENSCAD_AGENT> <query>\n"
            + "3.a OPENSCAD_AGENT can be used to program an OpenSCAD script for the user. The script will be compiled and rendered for the user."
            + "\n\nIf the user's prompt can be answered by one of these tools, Ditto will use it to answer the question. Otherwise, Ditto will answer the question itself.\n\n"
        )
        query = query_prefix + examples + "\n" + stmem_query
        return query

    def prompt(self, query, user_id="Human", face_name="none"):
        self.__create_load_memory(user_id=user_id)
        stamp = str(datetime.utcfromtimestamp(time.time()))

        mem_query = f"Timestamp: {stamp}\n{query}"

        if self.template_type == "DEFAULT":
            prompt = PromptTemplate(
                input_variables=["history", "input"], template=self.template
            )
            stmem_query = self.short_term_mem_store.get_prompt_with_stmem(
                user_id, query, face_name=face_name
            )
            conversation_with_memory = ConversationChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory[user_id],
                verbose=self.verbose,
            )
            res = conversation_with_memory.predict(input=stmem_query)
        else:
            stmem_query = self.short_term_mem_store.get_prompt_with_stmem(
                query, user_id, face_name=face_name
            )
            examples = self.example_store.get_example(query)
            query_with_examples = self.add_example_to_query(
                query, stmem_query, examples
            )
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=self.template,
                memory=self.memory[user_id],
            )
            conversation_with_memory = ConversationChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory[user_id],
                verbose=self.verbose,
            )

            log.info(f"Getting LLM response...")
            res = conversation_with_memory.predict(input=query_with_examples)

        if "GOOGLE_SEARCH" in res:  # handle google search
            log.info(f"Handling prompt for {user_id} with Google Search Agent")
            ditto_command = "<GOOGLE_SEARCH>"
            ditto_query = res.split("GOOGLE_SEARCH")[-1].strip()
            res = self.google_search_agent.handle_google_search(res)
            res = res + "\n-LLM Tools: Google Search-"
            memory_res = f"{ditto_command} {ditto_query} \nGoogle Search Agent: " + res

        elif (
            "PYTHON_AGENT" in res and user_id == self.llm_code_compiler_user
        ):  # handle llm code compiler
            log.info(f"Handling prompt for {user_id} with LLM Code Compiler")

            ditto_command = "<PYTHON_AGENT>"
            ditto_query = res.split("PYTHON_AGENT")[-1].strip()
            res = self.programmer_agent.prompt(ditto_query)
            res = res + "\n-LLM Tools: Programmer Agent-"
            memory_res = f"{ditto_command} {ditto_query} \nProgrammer Agent: " + res

            success = write_llm_code(
                res
            )  # TODO: might want to move this to assistant/ ( defininetly want to move this to assistant/)
            if success == "success":
                run_llm_code()

        elif "OPENSCAD_AGENT" in res:  # handle llm code compiler
            log.info(f"Handling prompt for {user_id} with LLM Code Compiler (OpenSCAD)")

            ditto_command = "<OPENSCAD_AGENT>"
            ditto_query = res.split("OPENSCAD_AGENT")[-1].strip()

            res = self.openscad_agent.prompt(ditto_query)
            res = res + "\n-LLM Tools: OpenSCAD Agent-"
            memory_res = f"{ditto_command} {ditto_query} \nOpenSCAD Agent: " + res

            run_openscad_code(res)

        else:  # no special handling
            memory_res = res

        self.save_new_memory(mem_query, memory_res, user_id, face_name=face_name)
        self.short_term_mem_store.save_response_to_stmem(user_id, query, memory_res)

        # save to knowledge graph (starts a new thread and closes when done)
        if self.kg_mode == True:
            kg_job = KGJob(
                user_id, str(query).replace('"', "'"), memory_res.replace('"', "'")
            )

        log.info(f"Handled prompt for {user_id}")
        return res


if __name__ == "__main__":
    ditto = DittoMemory(verbose=True)
