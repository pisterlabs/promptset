import os, sys

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

from .JuypyterClient import JupyterNotebook
from .BaseCodeInterpreter import BaseCodeInterpreter
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import tiktoken
from utils.utils import *

from openai import OpenAI
from retrying import retry
from dotenv import load_dotenv

load_dotenv()

from prompt.chatgpt_prompt import *


class GPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        super().__init__()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dialog = [
            {
                "role": "system",
                "content": CHATGPT_CODE_INTERPRETER_SYSTEM_PROMPT + "\n",
            },
        ]

        self.response = None

        PRE_EXEC_CODE_OUT = self.nb.add_and_run(PRE_EXEC_CODE)

        self.console = Console()  # for printing output
        self.stop_condition_met1, self.stop_condition_met2 = False, False

        # for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tiktoken.encoding_for_model(self.model)

        self.all_tok_in, self.all_tok_gen = 0, 0

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    def clean_the_dialog_one_step(self, question):
        question_idx = 0
        for idx, item in enumerate(self.dialog):
            if item["content"] == question:
                question_idx = idx

        filtered_dialog = self.dialog[question_idx:]

        user_qinit_dict = filtered_dialog[0]
        answer_fuse_str = "\n".join(
            [i["content"].strip() for i in filtered_dialog[1::2]]
        )

        self.dialog = self.dialog[:question_idx] + [
            {"role": "user", "content": user_qinit_dict["content"]},
            {"role": "assistant", "content": answer_fuse_str},
        ]

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def ChatCompletion(self, temperature: float = 0.1, top_p: float = 1.0):
        dialog_stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.dialog,
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024,
            stream=True,
        )

        self.all_tok_in += len(self.tokenizer.encode(dialog2rawtext(self.dialog)))

        stop_string1 = "```python"
        stop_string2 = "```"
        stop_max_len = max(len(stop_string1), len(stop_string2))
        self.stop_condition_met1, self.stop_condition_met2 = False, False
        in_code_block = False  # Flag to denote if we are inside a code block
        buffer = ""

        for chunk in dialog_stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                buffer += content  # add received content to the buffer

                if len(buffer) > stop_max_len:
                    buffer = buffer[-stop_max_len:]

                if stop_string1 in buffer:
                    self.stop_condition_met1 = True

                elif self.stop_condition_met1 and (stop_string2 in buffer):
                    self.stop_condition_met2 = True

                yield (
                    content,
                    (self.stop_condition_met1, self.stop_condition_met2),
                )  # yield received content

                # If the stop condition is met, break out of the loop
                if self.stop_condition_met2:
                    return

    def print_dialog(self):
        for dialog in self.dialog:
            role = dialog["role"]
            content = dialog["content"]

            if role.lower() == "system":
                self.console.print(Markdown(f"## SYSTEM_PROMPT\n\n{content}\n\n"))
            elif role.lower() == "user":
                self.console.print(Markdown(f"## 👤 User\n\n{content}\n"))
            elif role.lower() == "assistant":
                self.console.print(Markdown(f"## 🤖 Assistant\n\n{content}\n"))

    def chat(
        self,
        user_message: str,
        MAX_TRY: int = 6,
        temperature: float = 0.1,
        top_p: float = 1.0,
        VERBOSE: bool = True,
    ):
        if VERBOSE:
            self.console.print(Markdown(f"👤 User : **{user_message}**\n"))
            self.console.print(Markdown(f"🤖 Assistant :\n"))
        self.dialog.append({"role": "user", "content": user_message})

        # interactively and interatively code generation and refinement
        for i in range(MAX_TRY):
            generated_text_local = ""
            for char, cond in self.ChatCompletion(temperature=temperature, top_p=top_p):
                generated_text_local += char
                self.all_tok_gen += len(self.tokenizer.encode(char))  # for token count

                if VERBOSE:
                    if cond[0]:
                        self.console.print(char, style="code", end="")
                    elif cond[-1]:
                        self.console.print(char, end="")
                    else:
                        self.console.print(char, end="")

            # Get code block
            code_blocks = self.extract_code_blocks(generated_text_local)

            if code_blocks:
                code_output, error_flag = self.execute_code_and_return_output(
                    code_blocks[0]
                )

                response_content = f"{generated_text_local}\n```Execution Result:\n{code_output}\n```\n"
                if VERBOSE:
                    self.console.print(
                        f"\n```Execution Result:\n{code_output}\n```\n",
                        style="code",
                        end="",
                    )
                self.dialog.append({"role": "assistant", "content": response_content})
                self.dialog.append({"role": "user", "content": CHATGPT_FEEDBACK_PROMPT})

            else:
                if "<done>" in generated_text_local:
                    generated_text_local_refined = generated_text_local.split("<done>")[
                        0
                    ].strip()
                    self.dialog.append(
                        {"role": "assistant", "content": generated_text_local_refined}
                    )
                    break
                else:
                    self.dialog.append(
                        {"role": "assistant", "content": generated_text_local}
                    )
                    self.dialog.append({"role": "user", "content": "go ahead"})
                    continue

        # make all steps looks like an one answe
        self.clean_the_dialog_one_step(question=user_message)
        # self.print_dialog() # for debug

        if VERBOSE:
            self.console.print("\n")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("all_tok_in", style="dim", width=12)
            table.add_column("all_tok_gen", style="dim", width=12)
            table.add_row(str(self.all_tok_in), str(self.all_tok_gen))
            self.console.print(table)

        return self.dialog


if __name__ == "__main__":
    gpt_interpreter = GPTCodeInterpreter()

    answer = gpt_interpreter.chat(
        "Using the `numpy` library, generate a 3x3 matrix with random integer values between 1 and 5, then find its determinant. If the determinant is a prime number, square it."
    )

    gpt_interpreter.close()
