import re
import logging
from langchain import LLMChain
from langchain.schema import AIMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from caesura.phases.base_phase import ExecutionOutput, Phase
from caesura.phases.planning import PlanningPhase
from caesura.observations import ExecutionError, Observation, PlanFinished
from caesura.plan import ToolExecution, ToolExecutions
from caesura.utils import parse_args


logger = logging.getLogger(__name__)

class MappingPhase(Phase):

    def create_prompt(self, tools):
        result = ChatPromptTemplate.from_messages([
            self.system_prompt(tools),
            HumanMessagePromptTemplate.from_template(INIT_PROMPT),
        ])
        return result

    def system_prompt(self, tools):
        result = "You are Data-GPT, and you execute informal query plans using a set of tools:\n"
        result += self.database.describe()
        result += "You can use the following tools:\n"
        result += "\n".join([f"{t.name}: {t.description}" for t in tools])
        result += "\n" + FORMAT_INSTRUCTIONS
        return SystemMessagePromptTemplate.from_template(result)

    def init_chat(self, query, tools, plan, relevant_columns, step_nr, **kwargs):
        return self.create_prompt(tools).format_prompt(
            query=query, plan=str(plan), tool_names=", ".join([t.name for t in tools]),
            relevant_columns=relevant_columns.with_tool_hints(), plan_length=len(plan), step_nr=step_nr,
            step_prompt=plan[0].get_step_prompt()
        ).messages

    def execute(self, plan, step_nr, tools, chat_history, **kwargs):
        if step_nr > len(plan):
            raise PlanFinished

        prompt = ChatPromptTemplate.from_messages(chat_history)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        step = plan[step_nr - 1]

        ai_output = llm_chain.predict(stop=[f"Step {step_nr + 1}"])
        chat_history += [AIMessage(content=ai_output)]
        logger.info(ai_output)

        tool_calls = self.parse_tool_calls(ai_output, tools, step)
        step.set_tool_calls(tool_calls)
        logger.info(f"Running: Step {step_nr}: {step}")

        return ExecutionOutput(
            state_update={"plan": plan, "step": step},
            chat_history=chat_history,
        )

    def handle_observation(self, chat_history, observation, step_nr, plan, tools, query, **kwargs):
        if step_nr > len(plan):
            raise PlanFinished
        human_message_template = STEP_PROMPT if step_nr <= len(plan) else "Step {step_nr}: Finished!"
        if observation is not None and isinstance(observation, ExecutionError):
            self.handle_error(observation, chat_history, query, plan)
        elif observation is not None and isinstance(observation, Observation):
            msg = observation.get_message()
            if msg:
                chat_history.append(msg)
        step = plan[step_nr - 1]
        chat_history.append(HumanMessagePromptTemplate.from_template(human_message_template).format(
            step_nr=step_nr, step_prompt=step.get_step_prompt()))
        return chat_history
    
    def reinit_chat(self, observation, chat_history, plan, query, relevant_columns, tools, **kwargs):
        chat_history.append(observation.get_message(suffix="\nPlease restart from Step 1"))
        chat_history.append(HumanMessagePromptTemplate.from_template(INIT_PROMPT))
        chat_history = ChatPromptTemplate.from_messages(chat_history).format_prompt(
            query=query, plan=str(plan), tool_names=", ".join([t.name for t in tools]),
            relevant_columns=relevant_columns.with_tool_hints(), plan_length=len(plan), step_nr=1,
            step_prompt=plan[0].get_step_prompt()
        ).messages
        return chat_history

    def handle_error(self, error, chat_history, query, plan):
        error_tool = ""
        error_step = "one of the steps"
        if error.step_nr is not None:
            error_step = plan[error.step_nr - 1]
            error_tool = ", ".join([e.tool.name for e in error_step.tool_execs])
            error_tool = f"instead of {error_tool} "
            error_step = f"Step {error.step_nr}"
        msg = error.get_message(suffix= \
            "\nThis was my request: {query} and this is the plan I imagined: {plan}\n\n"\
            "To systematically fix this issue, answer the following questions one by one:\n"
            "1. What are potential causes for this error? Think step by step.\n"
            "2. Explain in detail how this error could be fixed.\n"
            "3. Is there a flaw in my plan (e.g. steps missing, wrong input table, ...) (Yes/No)?\n"
            "4. Is there a more suitable alternative plan (e.g. extracting information from image/text data instead of tabular metadata or vice versa) (Yes/No)?\n"
            "5. Should a different tool {error_tool}be selected for {error_step} (Yes / No)?\n"
            "6. Do the input arguments of some of the steps need to be updated (Yes / No)?\n"
        )
        prompt = ChatPromptTemplate.from_messages(chat_history + [msg])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answers = chain.predict(error_step=error_step, error_tool=error_tool, query=query, plan=plan.without_tools())
        logger.warning(error)
        logger.warning(answers)
        fix_idea, wrong_plan_str1, wrong_plan_str2, wrong_tool, wrong_input_args = \
            tuple(x.strip() for x in re.split(r"(^1\.|\n2\.|\n3\.|\n4\.|\n5\.|\n6\.)", answers)[slice(4, None, 2)])
        error.set_fix_idea(fix_idea)
        wrong_input_args, wrong_tool, wrong_plan1, wrong_plan2 = tuple(
            "yes" in re.split(r"\W", x.lower())
            for x in (wrong_input_args, wrong_tool, wrong_plan_str1, wrong_plan_str2)
        )
        if wrong_plan1 or wrong_plan2:
            if wrong_plan1:
                alternative_fix_idea = re.search(r"(^|\W)((Y|y)es)\W+(\w.+)\W", wrong_plan_str1)[4].strip(" .")
            if wrong_plan2:
                alternative_fix_idea = re.search(r"(^|\W)((Y|y)es)\W+(\w.+)\W", wrong_plan_str2)[4].strip(" .")

            if len(alternative_fix_idea) > 5:
                error.set_fix_idea(f"To fix the error: {alternative_fix_idea}.")
            error.set_target_phase(PlanningPhase)
            raise error
        if wrong_tool:
            error.set_target_phase(MappingPhase)
            raise error

        chat_history.append(error.get_message(suffix="\nPlease restart from Step 1"))
        return chat_history

    def parse_tool_calls(self, ai_out, tools, step):
        tools_str = [re.split(r"[,\.\n\(\:]", x.strip())[0].strip() for x in re.split("Tool(| [0-9]+):", ai_out)[2::2]]
        tool_map = {t.name: t for t in tools}
        found_separator = self.check_for_multiple_tools(tools_str)
        if found_separator:
            tools_str = [x.strip() for t in tools_str for x in t.split(found_separator)]
        args_str = [re.split("Tool(| [0-9]+):", x)[0].strip() for x in re.split("Arguments(| [0-9]+):", ai_out)[2::2]]
        if found_separator:
            args_str = [(x.strip() if x.strip().endswith(")") or not x.strip().startswith("(") else x.strip() + ")")
                        for a in args_str for x in a.split(")" + found_separator)]

        args_str = [" ".join([line.strip() for line in x.split("\n")]) for x in args_str]
        args_str = [parse_args(step, x) for x in args_str]

        tools_plan = [tool_map[x] for x in tools_str]
        tool_execs = ToolExecutions([ToolExecution(x, y) for x, y in zip(tools_plan, args_str)])
        return tool_execs

    def check_for_multiple_tools(self, tools_str):
        separators = (" and ", ",", ";")  # in case model decides to use more than one tool
        found_separator = None
        for t in tools_str:
            for s in separators:
                if s in t:
                    found_separator = s
                    break
            if found_separator is not None:
                break
        return found_separator


STEP_PROMPT = "Step {step_nr}: {step_prompt}"
INIT_PROMPT = "Execute the steps one by one. {relevant_columns}. Take these into account when executing the tools.\n" + \
    STEP_PROMPT


FORMAT_INSTRUCTIONS = (
    "Use the following output format:\n"
    "Step <i>: What to do in this step?\n"
    "Reasoning: Reason about which tool should be used for this step. Take datatypes into account.\n"
    "Tool: The tool to use, should be one of [{tool_names}]\n"
    "Arguments: The arguments to call the tool, separated by ';'. Should be (arg_1; ...; arg_n)\n"
    "(if you need more than one tool for this step, follow up with another Tool + Arguments.)"
)


# no improvements with these examples
#
# EXAMPLE_MAPPING = (
# """
# Example Mappings:
# Step X: Left Join the 'patient' and the 'patient_reports' table on the 'patient_id' column to combine the two tables.
# Reasoning: The SQL tool is the only tool that can join two tables.
# Tool: SQL
# Step X: Plot the 'result_table' in a bar plot. The 'diagnosis' should be on the X-axis and the 'mean_age' on the Y-Axis.
# Reasoning: Plots are generated using the 'Plot'  tool.
# Tool: Plot
# Step X: Select all rows of the 'pictures' table where the 'image' column depicts a skateboard, by looking the the images.
# Reasoning: In this step, rows need to be selected based on the content of images. Hence, the Image Select tool is appropriate.
# Tool: Image Select
# Step X: Look at the images in the 'image' column of the 'joined_table' table to determine the number of depicted persons.
# Reasoning: Looking at images and extracting information from images (number of depicted persons) requires the use of the Visual Question Answering tool.
# Tool: Visual Question Answering
# """
# )
