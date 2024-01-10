from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from utils.utils import *
import os, sys, json

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import io
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
import copy
from openai import OpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress


class MemoryBank:
    def __init__(self, MEMORY_DIR_PATH: str):
        # for printing
        self.console = Console()
        self.console.print(Markdown(f"## 🚀 Initializing MemoryBank..."))

        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)

        self.memory_dir_path = MEMORY_DIR_PATH

        if not os.path.exists(self.memory_dir_path):
            os.makedirs(self.memory_dir_path)

        # Load memories from dir
        self.memories = []
        self.load_memories()

        # memory organizer
        self.organizer = "gpt-4"

    def __len__(self):
        return len(self.memories)

    def update_memory_by_id(self, memory):
        # just linear search

        for idx, mem in enumerate(self.memories):
            if memory["id"] == mem["id"]:
                original_mem = copy.deepcopy(self.memories[idx])
                self.memories[idx] = {
                    "id": self.memories[idx]["id"],
                    "query": memory["query"],
                    "retrospection": memory["retrospection"],
                    "embedding_file": self.memories[idx]["embedding_file"],
                    "embedding": memory["embedding"],
                }
                self.console.print(
                    f"[bold]Memory Updated \n{original_mem} -> {self.memories[idx]}[/bold]"
                )
                return True

        return False

    def load_memories(self):
        json_files = [
            f
            for f in os.listdir(self.memory_dir_path)
            if f.endswith(".json") and ("memory_" in f)
        ]

        self.console.print(Markdown(f"## 📁 Loading memories from directory..."))

        for json_file in tqdm(json_files):
            with open(os.path.join(self.memory_dir_path, json_file), "r") as f:
                data = json.load(f)
                embedding_file = data["embedding_file"]
                with open(
                    os.path.join(self.memory_dir_path, embedding_file), "rb"
                ) as f:
                    buffer = f.read()
                    embedding = np.load(io.BytesIO(buffer))
                data["embedding"] = embedding
                self.memories.append(data)

        self.console.print(
            Markdown(f"## ✅ Successfully loaded {len(self.memories)} memory files!")
        )

    def save_memory(self, memory, embedding, memory_id: Optional[int] = None):
        if memory_id is None:
            memory_id = len(self.memories) + 1
        embedding_file = f"memory_{memory_id}_embedding.npy"
        np.save(os.path.join(self.memory_dir_path, embedding_file), embedding)

        memory["embedding_file"] = embedding_file
        json_file = f"memory_{memory_id}.json"
        with open(os.path.join(self.memory_dir_path, json_file), "w") as f:
            json.dump(memory, f)

    def encode(
        self,
        init_retrieved,
        retrospection_dict,
    ):
        if init_retrieved is None:
            self.console.print(f"No memory to encode (no retrieved memory exists)")
            return

        operation = retrospection_dict["action"]
        if isinstance(operation, str):
            operation = operation.lower()
        else:
            operation = "none"
        # if len(init_retrieved) == 0:
        #    operation = "<add>"  # when no retrieved memory it should be new
        if (retrospection_dict["title"] is None) or (
            len(retrospection_dict["title"]) == 0
        ):
            # query is too short (no content)
            return False
        query_embedding = get_embedding_ada(retrospection_dict["title"])

        # memory encoding by operation
        if "add" in operation:
            # adding new memory
            memory = {
                "id": len(self.memories) + 1,
                "query": retrospection_dict["title"],
                "retrospection": retrospection_dict["description"],
            }
            self.console.print(f"Memory Encoded : [bold][green]<Add>[/green][bold]")
            self.save_memory(memory, query_embedding)
            self.update_memory()
            return True
        elif "revise" in operation:
            # revise initial retrieved memory
            ## save revised memory (in disk)
            memory = {
                "id": init_retrieved[0]["id"],
                "query": retrospection_dict["title"],
                "retrospection": retrospection_dict["description"],
            }
            self.save_memory(
                memory,
                query_embedding,
                memory_id=init_retrieved[0]["id"],
            )
            ## update working memory (in RAM)
            self.console.print(f"Memory Encoded : [bold][green]<revised>[/green][bold]")
            memory["embedding"] = query_embedding
            self.update_memory_by_id(memory)
        else:
            # not permitted memory operation
            self.console.print(f"Memory Encoded : [bold][green]<none>[/green][bold]")
            return False

    def retrieve(self, query: str, retrieval_k: int = 3):
        # Get Top k similar memory
        embedding = get_embedding_ada(query)

        # If no memories yet
        if len(self.memories) == 0:
            self.console.print("No memories stored yet!")
            return []

        all_embeddings = np.vstack([memory["embedding"] for memory in self.memories])
        all_embeddings_normed = (
            all_embeddings / np.linalg.norm(all_embeddings, axis=1)[:, np.newaxis]
        )
        query_embedding_normed = embedding / np.linalg.norm(embedding)
        dot_products = np.dot(all_embeddings_normed, query_embedding_normed)
        top_k_indices = dot_products.argsort()[-retrieval_k:][::-1]

        # Retrieve the top-k memories
        retrieved_memories = [self.memories[i] for i in top_k_indices]

        return retrieved_memories

    def update_memory(self):
        json_files = [
            f
            for f in os.listdir(self.memory_dir_path)
            if f.endswith(".json") and ("memory_" in f)
        ]

        if len(self.memories) != len(json_files):
            memory_idx_to_load = list(
                range(len(self.memories) + 1, len(json_files) + 1)
            )
            for i in tqdm(memory_idx_to_load):
                with open(
                    os.path.join(self.memory_dir_path, f"memory_{i}.json"), "r"
                ) as f:
                    data = json.load(f)
                    embedding_file = data["embedding_file"]
                    with open(
                        os.path.join(self.memory_dir_path, embedding_file), "rb"
                    ) as f:
                        buffer = f.read()
                        embedding = np.load(io.BytesIO(buffer))
                    data["embedding"] = embedding
                    self.memories.append(data)

            self.console.print(f"✅ [green]Updated newly encoded memory...[green]")
            self.console.print(f"[blue]Total Memory : {len(self.memories)}[blue]")


class SelfDocAgent:
    def __init__(self, model: str = "gpt-4", memory: Optional[MemoryBank] = None):
        # print
        self.console = Console()

        # init model
        self.model = model
        self.llm_interpreter = GPTCodeInterpreter(model=model)
        self.INIT_SYS_PROMPT = self.llm_interpreter.dialog[0]["content"]

        # init memory
        self.MEMORY_DIR_PATH = os.path.join(
            prj_root_path, "memory", self.__class__.__name__, self.model
        )
        if memory is None:
            self.load_memory()
            self.console.print(Markdown(f"## New Memory Loaded\n"))
        else:
            self.memories = memory

        self.document = ""
        self.retrospect = ""

    def load_memory(self):
        self.memories = MemoryBank(MEMORY_DIR_PATH=self.MEMORY_DIR_PATH)
        return self.memories

    def get_memory(self):
        return self.memories

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm_interpreter.__exit__(exc_type, exc_value, traceback)

    def close(self):
        if not hasattr(self, "_is_closed") or not self._is_closed:
            self.llm_interpreter.close()
            self._is_closed = True

    def save_traj(self, traj_name: str):
        # check MEMORY_DIR_PATH exist
        if not os.path.exists(self.MEMORY_DIR_PATH):
            os.makedirs(self.MEMORY_DIR_PATH)

        save_dict = {
            "dialog": self.llm_interpreter.dialog,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": self.llm_interpreter.dialog[-1]["content"],
            "init_system_prompt": self.INIT_SYS_PROMPT,
            "retrieved": self.document,
            "retrospect": self.retrospect,
        }

        MEMORY_FILE_PATH = os.path.join(
            self.MEMORY_DIR_PATH,
            hashlib.sha1(self.instruction.encode()).hexdigest() + ".json"
            if traj_name == ""
            else traj_name,
        )

        # Save
        with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def get_retrospection(
        self,
        dialog: List[Dict],
        init_retrieved: Optional[List] = None,
        VERBOSE: bool = True,
        ORGANIZE_MEMORY: bool = True,
    ):
        full_traj = dialog_to_string(dialog, pretty=True)

        if (init_retrieved is None) or len(init_retrieved) <= 0:
            full_prompt = (
                f"<From prior experience>:\n\nNone\n\n"
                f"<Agent Log>:\n\n{full_traj}\n\n"
                # f"<Extra memories not chosen>:\n\nNone\n"
            )

        elif (init_retrieved is not None) and len(init_retrieved) >= 1:
            retrospect = init_retrieved[0]
            extra_mem = init_retrieved[1:]
            full_prompt = (
                f"<From prior experience>:\n\nTitle :\n{retrospect['query']}\n\nContent :\n{retrospect['retrospection']}\n\n"
                f"<Agent Log>:\n\n{full_traj}\n\n"
                # f"<Extra memories not chosen>:\n\n"
                # + "\n".join(
                #    f"[{index}]\n\nTitle: \n{memory['query']}\n\nContent :\n{memory['retrospection']}"
                #    for index, memory in enumerate(extra_mem)
                # )
            )

        else:
            raise f"Error during retrospection\ninit_retrieved : {init_retrieved}"

        retrospector = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = retrospector.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "As an AI specialist, your task is to create accurate and verified retrospection descriptions based on provided problem solutions. Your goal is to ensure that these retrospections are grounded in successfully executed solutions, serving as reliable guides for future problem-solving attempts.\n\nHere is you answer format : \n\n**Analyze Tool output**\nEnsure that the results from using the tool (python) align with the desired outcome.\n\n\n**Validation**: Assess if the Agent's log effectively addresses the user's queries based on the Analyze Tool Output. Verify the problem's resolution through the tool's output rather than theoretical assumptions.\n\n**Retrospection Draft** : If the tool's usage and results yield unexpected or different outcomes, consider if this should be documented for future reference. Include comparisons to prior experiences, highlighting what sets this new knowledge apart.\n\n**Action Decision**: Based on the validation, decide whether to:\n   - **Revise**  the new insight by incorporating it into an existing entry, ensuring that it is based on a verified and successful solution, or\n   - **Add** it as a new entry only if it represents a verified solution that provides new insights or approaches to problem-solving.\n   - **None** if no addition or revision is necessary.\n   \nIndicate your choice as:\n```action\nRevise, Add, or None\n```\n\n**Retrospection Crafting**:\n\na. Construct a succinct title that captures the core of the problem:\n```title\n(Concise Title)\n```\n\nb. Write a brief description, emphasizing the solution's unique aspects and any novel insights gained. This should be a succinct retrospection, capturing the essence of the solution.\ndescription:\n```description\n(Brief retrospection goes here)\n```\n\nYou must format the output with the action, title, and description wrapped in triple backticks.(```)"
                    if ORGANIZE_MEMORY
                    else "You are an AI specialist, tasked with crafting concise retrospections from provided problem solutions. Your aim is to distill the essence of the problem and its solution, capturing the lesson you learned.\n\n**Validation**: Your role here is to compare the current problem-solving approach, as described in the agent log, with the previously known solutions from prior experience. The objective is to determine if the current approach provides new knowledge or insights not present in prior experiences. Also, assess if the solution relates to or overlaps with the extra memories that were initially deemed unrelated. Determine if the agent found new, valuable information during the problem-solving process or if there was a critical turning point where failure eventually led to a solution.\n\n**Action Decision**: Based on the validation, always decide to:\n   - **Add** the new insight as a new, standalone entry.\n\nIndicate your choice as:\n```action\nAdd\n```\n\n**Retrospection Crafting**:\n\na. Construct a succinct title that captures the core of the problem:\n```title\n(Concise Title)\n```\n\nb. Articulate a brief description. This should highlight the unique aspects of the solution, emphasizing any newfound insights. \ndescription:\n```description\n(Brief retrospection goes here)\n```\n\nc. Since the decision is always to add as a new entry, indicate with `-1`.\n```location\n-1\n```\nYou must format the output with the action, title, description, and location wrapped in triple backticks.(```)",
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )
        del retrospector

        if VERBOSE:
            self.console.print(response.choices[0].message.content)
        if isinstance(response.choices[0].message.content, str):
            self.retrospect = response.choices[0].message.content
        extracted = self.extract_retrospect(response.choices[0].message.content)

        if VERBOSE:
            self.console.print(Markdown(f"## Full Retrospection :\n\n"))
            self.console.print(Markdown(f"### Memory Operation :\n\n"))
            self.console.print(Markdown(f"**{extracted['action']}**"))
            self.console.print(Markdown(f"### Title :\n\n"))
            self.console.print(Markdown(f"**{extracted['title']}**"))
            self.console.print(Markdown(f"### Retrospection :\n\n"))
            self.console.print(Markdown(f"**{extracted['description']}**"))

        return extracted

    def extract_retrospect(self, input_str: str):
        # Split by block delimiters
        blocks = {
            "action": None,
            "title": None,
            "description": None,
            "location": None,
        }

        # Extract 'action' content
        if "```action" in input_str:
            blocks["action"] = (
                input_str.split("```action")[1].split("```")[0].strip().lower()
            )

        # Extract 'title' content
        if "```title" in input_str:
            blocks["title"] = input_str.split("```title")[1].split("```")[0].strip()

        # Extract 'description' content
        if "```description" in input_str:
            desc_parts = input_str.split("```description")
            description_content = desc_parts[1]
            description_content = description_content.split("```location")[0].split(
                "```"
            )
            description_text = "```".join(description_content[:-1])

            blocks["description"] = description_text.strip()

        # Extract 'location' content
        if "```location" in input_str:
            blocks["location"] = (
                input_str.split("```location")[1].split("```")[0].strip()
            )

        return blocks

    def step(
        self,
        instruction: str,
        MAX_TRY: int = 7,
        temperature: float = 0.1,
        top_p: float = 0.95,
        VERBOSE: bool = True,
        USE_RETRIEVE: bool = True,
        USE_ENCODE: bool = True,
        ORGANIZE_MEMORY: bool = True,
        MULTITURN: bool = False,
    ) -> List[Dict]:
        self.instruction = instruction

        # step 1 : retrieve document first
        extra_sys_prompt = ""
        if USE_RETRIEVE:
            retrieved = self.memories.retrieve(self.instruction)
            if len(retrieved) >= 1:
                document = retrieved[0]["retrospection"]
                extra_sys_prompt = (
                    f"\nFrom prior experience you documented :\n{document}"
                )
                self.document = document
                if VERBOSE:
                    self.console.print(Markdown(f"## Document Retrieved\n\n{document}"))
            else:
                extra_sys_prompt = ""

        self.llm_interpreter.dialog[-1]["content"] = (
            self.INIT_SYS_PROMPT
            + extra_sys_prompt
            + "\n\nReflect on whether the past information is pertinent to the current question. Be cautious of interpreting and applying irrelevant information, as it might decrease accuracy. "
        )

        # step 2 : agent make trajectories
        self.llm_interpreter.chat(
            user_message=instruction,
            MAX_TRY=MAX_TRY,
            temperature=temperature,
            top_p=top_p,
            VERBOSE=VERBOSE,
        )
        if MULTITURN:
            second_round = input(" User >")
            self.llm_interpreter.chat(
                user_message=second_round,
                MAX_TRY=MAX_TRY,
                temperature=temperature,
                top_p=top_p,
                VERBOSE=VERBOSE,
            )

        # step3 : encode it's experience
        if USE_ENCODE:
            retrospection_dict = self.get_retrospection(
                dialog=self.llm_interpreter.dialog,
                init_retrieved=retrieved,
                VERBOSE=VERBOSE,
                ORGANIZE_MEMORY=ORGANIZE_MEMORY,
            )
            if ORGANIZE_MEMORY:
                self.memories.encode(
                    init_retrieved=retrieved,
                    retrospection_dict=retrospection_dict,
                )
            else:
                retrospection_dict["action"] = "add"
                self.memories.encode(
                    init_retrieved=retrieved,
                    retrospection_dict=retrospection_dict,
                )

        return self.llm_interpreter.dialog


if __name__ == "__main__":
    agent = SelfDocAgent()

    problem_example = """
Can you draw cat with stable diffusion xl?
```

"""

    agent.step(
        instruction=problem_example,
        USE_RETRIEVE=True,
        USE_ENCODE=True,
        ORGANIZE_MEMORY=True,
        VERBOSE=True,
        MULTITURN=False,
    )

    agent.save_traj("stablediffusion_xl_usage")
    agent.close()
