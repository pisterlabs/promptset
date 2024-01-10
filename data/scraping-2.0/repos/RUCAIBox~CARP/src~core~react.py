import re
import io
import signal
import json
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional
from collections import Counter
import timeout_decorator
import ast
import traceback
import sys
import openai
from typing import Any, Callable, List, Optional
from collections import Counter
import traceback


from src.core.runtime import GenericRuntime
from src.core.backend import call_gpt, call_chatgpt
from src.utils.utils import (
    extract_cot_answer,
    create_message,
    extract_ao_answer,
    create_message,
    extract_cot_answer,
    num_tokens_from_messages,
)
from src.utils.math_utils import compare_ans


class ReActInterface:
    def __init__(
        self,
        model: str = "code-davinci-002",
        runtime: Optional[Any] = None,
        stop: str = "\nOutput:",
        num_react_samples: int = 1,
        add_call_feedback: bool = False,
        temperature: float = 0,
        max_tokens: int = 512,
        reason_prompt_temp: str = None,
        reflect_prompt_temp: str = None,
        reason_bootstrap: dict = None,
        reflect_bootstrap: dict = None,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.max_steps = 15
        self.thought_prefix = "Thought:"
        self.action_prefix = "Action:"
        self.answer_prefix = "Final Answer:"
        self.output_prefix = "Output:"
        self.num_react_samples = num_react_samples
        self.add_call_feedback = add_call_feedback
        self.stop = stop
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reason_prompt_temp = reason_prompt_temp
        self.reflect_prompt_temp = reflect_prompt_temp
        self.reason_bootstrap = reason_bootstrap
        self.reflect_bootstrap = reflect_bootstrap

    def prompt_agent(self, prompt, stop=None, bootstrap=None, max_tokens=None):
        return call_gpt(
            prompt=prompt,
            bootstrap=bootstrap,
            model=self.model,
            stop=self.stop if stop is None else stop,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature,
            num_beams=1,
        )[0]

    def parse_thought_action(self, thought_action):
        thought, action = thought_action.split("\n")
        return thought, action

    def parse_action(self, text):
        func_call_str = text.strip().split(self.action_prefix)[-1].strip()
        func_call_str = re.search(r"\w+\(.*\)", func_call_str).group()
        func_call_str = func_call_str.replace("\\", "\\\\")
        func_call = ast.parse(func_call_str).body[0].value
        func_name = func_call.func.id
        func_args = [ast.literal_eval(arg) for arg in func_call.args]
        return func_call_str, func_name, func_args

    def update_scratchpad(self, step):
        self.single_scratchpad += step
        self.scratchpad += step

    def extract_final_answer(self, text):
        last_line = text.strip().replace("$", "").split(self.answer_prefix)[-1].strip()
        last_line = last_line.strip("。").strip(".").strip("：")
        return last_line

    def is_finish(self, action_name, output):
        if action_name == "finish":
            return True
        return False

    def clear_history(self):
        self.history = []

    @timeout_decorator.timeout(10)
    def execute(self, code: Optional[List[str]] = None):
        code = re.sub(r"([^\\])\\([a-zA-Z])", r"\1\\\\\2", code)
        try:
            return str(self.runtime.eval_code(code))
        except Exception as e:
            return f"[Error] {str(e)}"

    def get_action(self, gen):
        assert self.action_prefix in gen
        action, action_name, action_args = self.parse_action(gen)
        return action, action_name, action_args

    def get_output(self, action, action_name, action_args):
        def execute_action():
            result = self.execute(action)
            if result.startswith("[Error]") and not self.add_call_feedback:
                return ""
            return result

        self.update_scratchpad(f"\n{self.output_prefix}")
        if action_name in self.runtime.GLOBAL_DICT:
            # execute action
            output = execute_action()
            if len(output) != 0:
                return output

        output = self.prompt_agent(
            self.build_reason_prompt(), stop="\n", bootstrap=self.reason_bootstrap
        )
        return output

    def build_reason_prompt(self):
        reason_prompt = self.reason_prompt_temp.replace("{question}", self.question)
        reason_prompt = reason_prompt.replace("{scratchpad}", self.scratchpad)
        return reason_prompt

    def extract_last_action(self):
        text = self.last_scratchpad
        last_line = text.strip().replace("$", "").split(self.output_prefix)[-1].strip()
        last_line = last_line.strip("。").strip(".")
        if "[" in last_line:
            try:
                output_list = eval(last_line)
                if isinstance(output_list, list) and len(output_list) == 1:
                    return output_list[0]
            except:
                return last_line
        return last_line

    def reason_step(self):
        self.last_scratchpad = self.single_scratchpad
        self.single_scratchpad = ""
        self.update_scratchpad("\n")
        if "finish" in self.reason_prompt_temp:
            self.update_scratchpad("Action: ")
        reason_prompt = self.build_reason_prompt()
        # action
        action = self.prompt_agent(reason_prompt, bootstrap=self.reason_bootstrap)
        if self.answer_prefix not in action and self.action_prefix not in action:
            return self.extract_last_action()
        if "Question" in action:
            return self.extract_last_action()
        if "End" in action or "end()" in action or "解题过程" in action or "问题" in action:
            return self.extract_last_action()
        self.update_scratchpad(action)
        if "Final Answer" in action:
            return self.extract_final_answer(action)
        try:
            action, action_name, action_args = self.parse_action(action)
        except:
            action, action_name, action_args = None, None, None
        output = self.get_output(action, action_name, action_args)
        self.update_scratchpad(" " + output.replace("  ", " "))
        if self.is_finish(action_name, output):
            return output
        return None

    def run(self, question):
        self.question = question
        self.history = []
        self.answers = []
        for _ in range(self.num_react_samples):
            self.scratchpad = ""
            self.single_scratchpad = ""
            self.contexts = []
            self.outputs = []
            answer = None
            for _ in range(self.max_steps):
                try:
                    answer = self.reason_step()
                except Exception as e:
                    print(traceback.format_exc())
                    print("Step 存在错误")
                    answer = None
                if answer is not None:
                    break
            self.answers.append(answer if answer is not None else "-10000")
            self.history.append(self.scratchpad)
        counter = Counter(self.answers)
        return counter.most_common(1)[0][0]


class ChatReActInterface(ReActInterface):
    def __init__(
        self,
        model: str = "code-davinci-002",
        runtime: GenericRuntime = None,
        stop: str = "\nOutput:",
        num_react_samples: int = 1,
        add_call_feedback: bool = False,
        temperature: float = 0,
        max_tokens: int = 512,
        reason_prompt_temp: str = None,
        reflect_prompt_temp: str = None,
        reason_bootstrap: dict = None,
        reflect_bootstrap: dict = None,
        drop_system: bool = True,
        do_reverse_role: bool = True,
        case_spliter: str = None,
    ) -> None:
        super().__init__(
            model,
            runtime,
            stop,
            num_react_samples,
            add_call_feedback,
            temperature,
            max_tokens,
            reason_prompt_temp,
            reflect_prompt_temp,
            reason_bootstrap,
            reflect_bootstrap,
        )
        self.drop_system = drop_system
        self.do_reverse_role = do_reverse_role
        self.case_spliter = case_spliter

    def build_reason_examples(self):
        # Initialize prompt, including description and sample, sample here is cut by `//``, Question and Output to user, Action to assistant
        system_role = "system" if not self.drop_system else "user"
        self.prompt = [create_message(system_role, self.reason_bootstrap)]
        case_spliter = (
            "\n\n" if self.case_spliter is None else f"\n{self.case_spliter}\n"
        )
        self.examples = self.reason_prompt_temp.split(case_spliter)
        line_spliter = "\n" if "//" not in self.reason_prompt_temp else "\n//\n"
        for e in self.examples:
            messages = e.split(line_spliter)
            for m in messages:
                if m.startswith("Question") or m.startswith("Output"):
                    self.prompt.append(create_message("user", m))
                else:
                    self.prompt.append(create_message("assistant", m))
        # self.scratchpad = [create_message("user", f"Question: {self.question}")]
        self.prompt.append(create_message("user", f"Question: {self.question}"))

    def update_scratchpad(self, step):
        if (
            step.startswith("Question")
            or step.startswith("Output")
            or step.startswith("Trial")
        ):
            self.scratchpad.append(create_message("user", step))
        else:
            self.scratchpad.append(create_message("assistant", step))

    def get_completion_prompt(self):
        prompt = f"{self.reason_bootstrap}\n\n{self.reason_prompt_temp}\n\nQuestion: {self.question}\n"
        prompt = prompt.replace("\n//\n", "\n")
        steps = [s["content"] for s in self.scratchpad]
        prompt += "\n".join(steps)
        return prompt

    def prompt_chat_agent(self, prompt=None, stop=None, max_tokens=None):
        if prompt is not None:
            pass
            # print(prompt)
        else:
            self.build_reason_examples()
            # print(self.prompt + self.scratchpad)
        return call_chatgpt(
            self.prompt + self.scratchpad if prompt is None else prompt,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=stop if stop is not None else self.stop,
            temperature=0.0,
            num_beams=1,
        )[0]

    def reverse_chat_role(self):
        def reverse(role):
            if role == "user":
                return "assistant"
            if role == "assistant":
                return "user"
            return role

        rprompts = [self.prompt[0].copy()]
        for p in self.prompt[1:]:
            rp = p.copy()
            rp["role"] = reverse(p["role"])
            rprompts.append(rp)
        for p in self.scratchpad:
            rp = p.copy()
            rp["role"] = reverse(p["role"])
            rprompts.append(rp)
        return rprompts

    def get_output(self, action, action_name, action_args):
        def execute_action():
            result = self.execute(action)
            if result.startswith("[Error]") and not self.add_call_feedback:
                return ""
            return result

        if action_name in self.runtime.GLOBAL_DICT:
            output = execute_action()
            if len(output) != 0:
                return output

        if not self.do_reverse_role:
            # If the model needs to generate its own output, switch back to the previous method
            prompt = self.get_completion_prompt()
            prompt += f"\n{self.output_prefix}"
            print("output", repr(prompt))
            output = self.prompt_agent(prompt, stop="\n")
        else:
            # Construct a prompt that transposes user and assitant
            self.build_reason_examples()
            prompt = self.reverse_chat_role()
            output = self.prompt_chat_agent(prompt)
        return output

    def extract_last_action(self):
        text = None
        contents = []
        for s in self.scratchpad[::-1]:
            contents.append(s["content"])
            if s["content"].startswith(self.output_prefix):
                text = s["content"]
                break
        if text is None:
            text = "\n".join(contents)
        last_line = text.strip().split(self.output_prefix)[-1].strip()
        last_line = last_line.strip("。").strip(".")
        if "[" in last_line:
            try:
                output_list = eval(last_line)
                if isinstance(output_list, list) and len(output_list) == 1:
                    return output_list[0]
            except:
                return last_line
        return last_line

    def reason_step(self):
        # Get Action
        action = self.prompt_chat_agent()
        # print("action", action)
        # Judge whether stop
        if self.answer_prefix not in action and self.action_prefix not in action:
            return self.extract_last_action()
        if "Question" in action:
            return self.extract_last_action()
        if "End" in action or "end()" in action or "解题过程" in action or "问题" in action:
            return self.extract_last_action()
        self.update_scratchpad(action)
        if "Final Answer" in action:
            return extract_ao_answer(action)
        # Parse Action
        try:
            action, action_name, action_args = self.parse_action(action)
        except:
            action, action_name, action_args = None, None, None
        # Get Output
        output = self.get_output(action, action_name, action_args)
        if not output.startswith(self.output_prefix):
            self.update_scratchpad(f"{self.output_prefix} {output}")
        else:
            self.update_scratchpad(output)
        # Judge whether stop
        if self.is_finish(action_name, output):
            return output
        return None

    def run(self, question):
        self.question = question
        self.history = []
        self.answers = []
        for _ in range(self.num_react_samples):
            self.scratchpad = []
            self.build_reason_examples()
            answer = None
            for _ in range(self.max_steps):
                try:
                    answer = self.reason_step()
                except Exception as e:
                    print(traceback.format_exc())
                    print("Step 存在错误")
                    answer = None
                if answer is not None:
                    break
            self.answers.append(answer if answer is not None else "-10000")
            self.history.append(self.scratchpad)
        counter = Counter(self.answers)
        return counter.most_common(1)[0][0]


class ChatRewriteReActInterface(ChatReActInterface):
    def __init__(
        self,
        model: str = "code-davinci-002",
        runtime: GenericRuntime = None,
        stop: str = "\nOutput:",
        num_react_samples: int = 1,
        add_call_feedback: bool = False,
        temperature: float = 0,
        max_tokens: int = 512,
        reason_prompt_temp: str = None,
        reflect_prompt_temp: str = None,
        reason_bootstrap: dict = None,
        reflect_bootstrap: dict = None,
        drop_system: bool = True,
        do_reverse_role: bool = True,
        do_drop_nl_answer: bool = False,
        nl_answer_prefix: str = "答案是",
        case_spliter: str = None,
    ) -> None:
        super().__init__(
            model,
            runtime,
            stop,
            num_react_samples,
            add_call_feedback,
            temperature,
            max_tokens,
            reason_prompt_temp,
            reflect_prompt_temp,
            reason_bootstrap,
            reflect_bootstrap,
            drop_system,
            do_reverse_role,
            case_spliter,
        )
        self.do_drop_nl_answer = do_drop_nl_answer
        self.nl_answer_prefix = nl_answer_prefix

    def build_reason_examples(self):
        # Initialize prompt
        system_role = "system" if not self.drop_system else "user"
        self.prompt = [create_message(system_role, self.reason_bootstrap)]
        case_spliter = (
            "\n\n\n" if self.case_spliter is None else f"\n{self.case_spliter}\n"
        )
        self.examples = self.reason_prompt_temp.split(case_spliter)
        if self.reason_prompt_temp.startswith("Trial"):
            last_prompt = create_message(
                "user", f"Trial: {self.old_scratchpad}\nQuestion: {self.question}"
            )
        else:
            last_prompt = create_message(
                "user", f"Question: {self.question}\nTrial: {self.old_scratchpad}"
            )
        max_tokens = 512
        for e in self.examples:
            messages = e.split("\n//\n")
            example_prompt = []
            for m in messages:
                if (
                    m.startswith("Question")
                    or m.startswith("Output")
                    or m.startswith("Trial")
                ):
                    # if m.startswith("Question") or m.startswith("Output"):
                    example_prompt.append(create_message("user", m))
                else:
                    example_prompt.append(create_message("assistant", m))
            num_tokens = (
                num_tokens_from_messages(self.prompt + example_prompt + [last_prompt])
                + max_tokens
            )
            print("num_tokens", num_tokens)
            if num_tokens >= 4090:
                print("[Warning] skip reason example")
                break
            self.prompt += example_prompt
        self.prompt.append(last_prompt)

    def drop_nl_final_answer(self, text):
        return text.strip().split(self.nl_answer_prefix)[0].strip()

    def run(self, question, old_scratchpad):
        if self.do_drop_nl_answer:
            old_scratchpad = self.drop_nl_final_answer(old_scratchpad)
        self.old_scratchpad = old_scratchpad
        return super().run(question)


class ChatIterRewriteRetActionInterface(ChatRewriteReActInterface):
    def __init__(
        self,
        model: str = "code-davinci-002",
        runtime: GenericRuntime = None,
        stop: str = "\nOutput:",
        num_react_samples: int = 1,
        add_call_feedback: bool = False,
        temperature: float = 0,
        max_tokens: int = 512,
        reason_prompt_temp: str = None,
        reflect_prompt_temp: str = None,
        reason_bootstrap: dict = None,
        reflect_bootstrap: dict = None,
        drop_system: bool = True,
        do_reverse_role: bool = True,
        do_drop_nl_answer: bool = False,
        case_spliter: str = None,
        num_trials: int = 3,
        nl_answer_prefix: str = "答案是",
        question_prefix: str = "Question: ",
        message_schema: str = None,
        reflect_max_tokens: int = 512,
    ) -> None:
        super().__init__(
            model,
            runtime,
            stop,
            num_react_samples,
            add_call_feedback,
            temperature,
            max_tokens,
            reason_prompt_temp,
            reflect_prompt_temp,
            reason_bootstrap,
            reflect_bootstrap,
            drop_system,
            do_reverse_role,
            do_drop_nl_answer,
            nl_answer_prefix,
            case_spliter,
        )
        self.num_trials = num_trials
        self.question_prefix = question_prefix
        self.message_schema = message_schema
        self.reflect_max_tokens = reflect_max_tokens

    def build_reflect_prompt(self):
        # Initialize Stage 2 prompt
        system_role = "system" if not self.drop_system else "user"
        self.reflect_prompt = [create_message(system_role, self.reflect_bootstrap)]
        case_spliter = (
            "\n\n\n" if self.case_spliter is None else f"\n{self.case_spliter}\n"
        )
        examples = self.reflect_prompt_temp.split(case_spliter)
        max_tokens = (
            self.reflect_max_tokens if self.reflect_max_tokens is not None else 512
        )
        for e in examples:
            messages = e.split("\n//\n")
            example_prompt = []
            for m in messages:
                if (
                    m.startswith("Question")
                    or m.startswith(self.output_prefix)
                    or m.startswith("问题")
                ):
                    example_prompt.append(create_message("user", m))
                else:
                    example_prompt.append(create_message("assistant", m))
            num_tokens = (
                num_tokens_from_messages(
                    self.reflect_prompt + example_prompt + self.get_scratchpad()
                )
                + max_tokens
            )
            if num_tokens >= 4090:
                print("[Warning] skip reflect example")
                break
            self.reflect_prompt += example_prompt

    def build_reason_examples(self):
        # Initialize Stage 1 prompt
        system_role = "system" if not self.drop_system else "user"
        self.prompt = [create_message(system_role, self.reason_bootstrap)]
        case_spliter = (
            "\n\n\n" if self.case_spliter is None else f"\n{self.case_spliter}\n"
        )
        self.examples = self.reason_prompt_temp.split(case_spliter)
        if self.reason_prompt_temp.startswith("Trial"):
            last_prompt = create_message(
                "user", f"Trial: {self.old_scratchpad}\nQuestion: {self.question}"
            )
        else:
            last_prompt = create_message(
                "user", f"Question: {self.question}\nTrial: {self.old_scratchpad}"
            )
        max_tokens = self.reflect_max_tokens
        for e in self.examples:
            messages = e.split("\n//\n")
            example_prompt = []
            for m in messages:
                if (
                    m.startswith("Question")
                    or m.startswith("Output")
                    or m.startswith("Trial")
                ):
                    # if m.startswith("Question") or m.startswith("Output"):
                    example_prompt.append(create_message("user", m))
                else:
                    example_prompt.append(create_message("assistant", m))
            num_tokens = (
                num_tokens_from_messages(
                    self.prompt + example_prompt + [last_prompt] + self.scratchpad
                )
                + max_tokens
            )
            if num_tokens >= 4090:
                print("[Warning] skip reason example")
                break
            self.prompt += example_prompt
        self.prompt.append(last_prompt)

    def get_scratchpad(self):
        def build_user_message():
            message = self.message_schema.replace("{question}", self.question)
            message = message.replace("{old_scratchpad}", self.old_scratchpad)
            message = message.replace("{scratchpad}", scratchpad)
            return message

        scratchpad = "\n".join([s["content"] for s in self.scratchpad])
        if self.message_schema is None:
            scratchpad = [
                create_message(
                    "user",
                    f"问题:\n{self.question}\n\n已有解题过程:\n{self.old_scratchpad}\n\n检验:\n{scratchpad}\n\n根据检验修正已有解题过程中的计算错误:",
                )
            ]
        else:
            scratchpad = [create_message("user", build_user_message())]
        return scratchpad

    def reflect_step(self):
        scratchpad = self.get_scratchpad()
        reflect = self.prompt_chat_agent(
            self.reflect_prompt + scratchpad, max_tokens=self.reflect_max_tokens
        )
        return reflect

    def reason_run(self):
        self.scratchpad = []
        self.build_reason_examples()
        answer = None
        for _ in range(self.max_steps):
            try:
                answer = self.reason_step()
            except Exception as e:
                print(traceback.format_exc())
                print("Step 存在错误")
                answer = None
            if answer is not None:
                break
        answer = answer if answer is not None else "-10000"
        return answer

    def remove_final_answer(self, text):
        return text.strip().split(self.nl_answer_prefix)[0].strip()

    def run(self, question, old_scratchpad):
        self.question = question
        self.nl_answers = []
        self.com_answers = []
        self.history = []
        self.nl_history = []
        self.scratchpad = []
        self.old_scratchpad = old_scratchpad  # TODO handle question
        # Extract cot answer
        nl_answer = extract_cot_answer(self.old_scratchpad, self.nl_answer_prefix)
        self.nl_answers.append(nl_answer)
        self.nl_history.append(self.old_scratchpad)
        for _ in range(self.num_trials):
            # cot -> react
            com_answer = self.reason_run()
            self.com_answers.append(com_answer)
            is_com_same = len(self.history) > 0 and self.history[-1] == self.scratchpad
            self.history.append(self.scratchpad)
            self.old_com_scratchpad = self.scratchpad
            # react -> cot
            self.build_reflect_prompt()
            self.old_scratchpad = self.reflect_step()
            is_nl_same = (
                len(self.nl_history) > 0 and self.nl_history[-1] == self.old_scratchpad
            )
            self.nl_history.append(self.old_scratchpad)
            nl_answer = extract_cot_answer(self.old_scratchpad, self.nl_answer_prefix)
            self.nl_answers.append(nl_answer)
            # print("react", com_answer)
            # print("compare2", com_answer, nl_answer)
            # Stopping criteria
            if compare_ans(nl_answer, com_answer):
                return com_answer
            if is_com_same and is_nl_same:
                break
        # Extract last ReAct answer
        return self.com_answers[-1]
