from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (AIMessage, HumanMessagePromptTemplate,
                                    SystemMessage)

from src.configure import (internal_prompt, template_codefix_data,
                           template_codefix_execution, template_codegen,
                           template_formulation)
from src.utils import get_solver_demo, get_solver_instruction


class Messages:
    def __init__(self, problem) -> None:
        self.global_conversations = []
        self.problem = problem
        self.system_message = SystemMessage(content=internal_prompt)

        self._formulation_response = None

    def user_says_header(self):
        self.global_conversations.append("\n-------")
        self.global_conversations.append("User says: ")
        self.global_conversations.append("---------\n")

    def chatbot_says_header(self, model):
        self.global_conversations.append("\n----------")
        self.global_conversations.append(f"{model} says: ")
        self.global_conversations.append("------------\n")

    def prompt_format(self, template):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=template
        ).format_messages(
            PROBLEM_TYPE=self.problem.data["problem_type"],
            PROBLEM_INFO=self.problem.data["problem_info"],
            INPUT_FORMAT=self.problem.data["input_format"],
            OBJECTIVE=self.problem.data["objective_info"],
            OUTPUT_INFO=self.problem.data["output_info"],
            OUTPUT_FORMAT=self.problem.data["output_format"],
            INITIAL_TEST_SCRIPT=self.problem.data["initial_test_script"],
            CODE=self.problem.data["code"],
            CODE_AVAILABLE=self.problem.data["code_available"],
            SOLVER=self.problem.solver,
            SOLVER_INSTRUCTION=get_solver_instruction(self.problem.solver),
            SOLVER_VAR_DEMO=get_solver_demo(self.problem.solver)["var"],
            SOLVER_CONSTR_DEMO=get_solver_demo(self.problem.solver)["constr"],
            SOLVER_SOLVE_DEMO=get_solver_demo(self.problem.solver)["solve"],
            ERROR_MESSAGE=self.problem.data["error"],
        )
        return chat_prompt_template

    def model_format(self, template):
        # this is used only for logging
        formatted_template = template.format(
            PROBLEM_TYPE=self.problem.data["problem_type"],
            PROBLEM_INFO=self.problem.data["problem_info"],
            INPUT_FORMAT=self.problem.data["input_format"],
            OBJECTIVE=self.problem.data["objective_info"],
            OUTPUT_INFO=self.problem.data["output_info"],
            OUTPUT_FORMAT=self.problem.data["output_format"],
            INITIAL_TEST_SCRIPT=self.problem.data["initial_test_script"],
            CODE=self.problem.data["code"],
            CODE_AVAILABLE=self.problem.data["code_available"],
            SOLVER="gurobi",
            SOLVER_INSTRUCTION=get_solver_instruction(self.problem.solver),
            SOLVER_VAR_DEMO=get_solver_demo(self.problem.solver)["var"],
            SOLVER_CONSTR_DEMO=get_solver_demo(self.problem.solver)["constr"],
            SOLVER_SOLVE_DEMO=get_solver_demo(self.problem.solver)["solve"],
            ERROR_MESSAGE=self.problem.errmsg,
        )
        return formatted_template

    def get_formulation_conversation(self):
        formumation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )

        conversation = [self.system_message, formumation_request]
        messages = self.prompt_format(conversation)

        # this is just to update conversation history
        # add update method
        self.user_says_header()
        self.global_conversations.append(self.system_message.content)
        self.global_conversations.append(self.model_format(template_formulation))
        self.chatbot_says_header("gpt-3.5-turbo")

        return messages

    def get_code_conversation(self):
        assert self.formulation_response is not None
        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        formulation_response = AIMessage(content=self.formulation_response)
        condegen_request = HumanMessagePromptTemplate.from_template(template_codegen)

        conversations = [
            self.system_message,
            formulation_request,
            formulation_response,
            condegen_request,
        ]

        messages = self.prompt_format(conversations)

        self.user_says_header()
        self.global_conversations.append(self.system_message.content)
        self.global_conversations.append(self.model_format(template_codegen))
        self.chatbot_says_header("gpt-3.5-turbo")

        return messages

    def get_code_fix_conversation(self, execution_ok):
        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        formulation_response = AIMessage(content=self.formulation_response)
        codefix_execution_request = HumanMessagePromptTemplate.from_template(
            template_codefix_execution
        )
        codefix_data_request = HumanMessagePromptTemplate.from_template(
            template_codefix_data
        )

        if not execution_ok:
            codefix_request = codefix_execution_request
            template_codefix = template_codefix_execution
        else:
            codefix_request = codefix_data_request
            template_codefix = template_codefix_data

        conversations = [
            self.system_message,
            formulation_request,
            formulation_response,
            codefix_request,
        ]

        messages = self.prompt_format(conversations)

        self.user_says_header()
        self.global_conversations.append(self.model_format(template_codefix))

        self.chatbot_says_header("gpt-3.5-turbo")

        return messages

    @property
    def formulation_response(self):
        return self._formulation_response

    @formulation_response.setter
    def formulation_response(self, llm_response: str):
        self._formulation_response = llm_response

    @property
    def path_to_conversation(self):
        return f"{self.problem.problem_path}/description.log"
