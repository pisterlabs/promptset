import ast
from typing import List

from langchain.schema import SystemMessage, HumanMessage, AIMessage

from autocoder.ai import AI
from autocoder.chat import print_autocoder_msg, ask_approval, prompt
from autocoder.project.project import Project
from autocoder.prompts.planning import QUESTIONS_PROMPT, PLAN_PROMPT
from autocoder.prompts.system import project_prompt


class Planner:
    def __init__(self, ai: AI, project: Project):
        self.ai = ai
        self.project = project
        self.chat = None

    def run(self, task: str) -> str:
        self.chat = self.init_messages(task)
        return self.generate_plan()

    def init_messages(self, task):
        files = self.project.read_all_files()
        files_str = "\n".join(str(file) for file in files)

        source_code_path = self.project.source_code_path
        tests_path = self.project.tests_path

        messages = [SystemMessage(content=project_prompt.format(files=files_str, source_code_path=source_code_path,
                                                                tests_path=tests_path)),
                    HumanMessage(content=f"Task: {task}")]

        return messages

    def generate_plan(self) -> str:
        plan = None
        approved = False
        while not approved:
            questions = self.generate_questions()
            if questions:
                print_autocoder_msg("Before we proceed there are a few things I need to clarify")
                self.ask_questions(questions)

            plan = self.generate_plan_from_chat()

            plan_msg = f"Here's what I propose\n\n{plan}"
            self.chat.append(AIMessage(content=plan_msg))
            print_autocoder_msg(plan_msg)

            ask_approval_msg = "Do you approve this plan?"
            self.chat.append(AIMessage(content=ask_approval_msg))
            approved = ask_approval(ask_approval_msg)

            if not approved:
                ask_feedback_msg = "What would you like to add?"
                self.chat.extend([AIMessage(content=ask_feedback_msg),
                                  HumanMessage(content=prompt(ask_feedback_msg))])

        return plan

    def generate_questions(self) -> List[str]:
        self.chat.append(SystemMessage(content=QUESTIONS_PROMPT))
        print_autocoder_msg("Thinking... :thinking_face:")
        questions_str = self.ai.call(self.chat)
        return ast.literal_eval(questions_str)

    def generate_plan_from_chat(self) -> str:
        self.chat.append(SystemMessage(content=PLAN_PROMPT))
        print_autocoder_msg("Thinking... :thinking_face:")
        return self.ai.call(self.chat)

    def ask_questions(self, questions: List[str]) -> None:
        while questions:
            q = questions.pop(0)
            self.chat.append(AIMessage(content=q))
            answer = prompt(q)
            self.chat.append(HumanMessage(content=answer))
