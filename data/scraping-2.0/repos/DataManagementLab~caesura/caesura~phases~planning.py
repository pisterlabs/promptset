import logging
import re
from langchain import LLMChain
from caesura.capabilities import ALL_CAPABILITIES
from langchain.schema import AIMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate

from caesura.phases.base_phase import ExecutionOutput, Phase
from caesura.phases.discovery import DiscoveryPhase
from caesura.observations import ExecutionError
from caesura.plan import Plan, PlanStep


logger = logging.getLogger(__name__)

INIT_PROMPT = "My request is: {query}. {relevant_columns}."
REINIT_PROMPT = "{relevant_columns} Take these updated relevant values into account when fixing the error!"


EXAMPLE_PLAN = """
Example plan for database containing 'patient' and 'patient_reports' tables:
Request: Plot the average patient age for each diagnosis.
Step 1: Join the 'patient' and the 'patient_reports' table on the 'patient_id' column to combine the two tables.
Input: patient, patient_reports
Output: joined_patient_table
New Columns: N/A
Step 2: Extract the diagnosis for each patient from the 'report' column in the 'joined_patient_table' table.
Input: joined_patient_table
Output: N/A
New Columns: diagnosis
Step 3: Group the 'joined_patient_table' by diagnosis and aggregate the 'age' column using the mean.
Input: joined_patient_table
Output: result_table
New Columns: mean_age
Step 4: Plot the 'result_table' in a bar plot. The 'diagnosis' should be on the X-axis and the 'mean_age' on the Y-Axis.
Input result_table
Output: N/A
New Columns: N/A
Step 5: Plan completed.

Example plan for database containing 'pictures' and a 'metadata' table:
Request: Get the number of pictures that depict a skateboard per epoch.
Step 1: Select all rows of the 'pictures' table where the 'image' column depicts a skateboard.
Input: pictures
Output: skateboard_pictures
New Columns: N/A
Step 2: Join the 'pictures' and the 'metadata' tables on the 'picture_path' column.
Input: pictures, metadata
Output: joined_table
New Columns: N/A
Step 3: Group by the 'epoch' column and count the number of rows.
Input: joined_table
Output: result
New Columns: num_pictures
Step 4: Plan completed.

Example plan for database containing tables 'building' and 'surveillance':
Request: Construct a table containing the highest number of persons depicted in a surveillance image per building.
Step 1: Join the 'building' table with the 'surveillance' table on the 'building_id' column.
Input: building, surveillance
Output: joined_table
New Columns: N/A
Step 2: Extract the number of depicted persons for each image in the 'image' column of the 'joined_table' table.
Input: joined_table
Output: N/A
New Columns: num_depicted_persons
Step 3: Group the 'joined_table' table by 'building_name' and aggregate the 'num_depicted_persons' column using the maximum.
Input: joined_table
Output: final_result
New Columns: max_num_depicted_persons
Step 5: Plan completed.

Example plan for chess database for a chess tournament. It has two tables 'chess_game_reports' and 'participating_players'.
Request: What is the highest number of moves in a chess game for each player.
Step 1: Join the 'chess_game_reports' and the 'participating_players' on the 'player_id' column.
Input: chess_game_reports, participating_players
Output: joined_table
New Columns: N/A
Step 2: Extract the number of moves from the chess game reports.
Input: joined_table
Output: N/A
New Columns: num_moves
Step 3: Group by 'player_name' and compute the maximum of 'num_moves'.
Input: joined_table
Output: result_table
New Columns: max_num_moves
Step 4: Plan completed.
"""


class PlanningPhase(Phase):

    def create_prompt(self):
        result = ChatPromptTemplate.from_messages([
            self.system_prompt(),
            HumanMessagePromptTemplate.from_template(INIT_PROMPT),
            AIMessagePromptTemplate.from_template("Request: {query}\nThought:")
        ])
        return result

    def system_prompt(self):
        result = "You are Data-GPT and you generate plans to retrieve data from databases:\n"
        result += EXAMPLE_PLAN + "\n\n"
        result += self.database.describe()
        result += "You have the following capabilities:\n"
        result += "\n".join([c.description for c in ALL_CAPABILITIES])
        result += "\n" + FORMAT_INSTRUCTIONS
        return SystemMessagePromptTemplate.from_template(result)

    def init_chat(self, query, relevant_columns, **kwargs):
        return self.create_prompt().format_prompt(query=query, relevant_columns=relevant_columns.with_join_hints()).messages

    def execute(self, chat_history, **kwargs):
        prompt = ChatPromptTemplate.from_messages(chat_history)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        ai_output = llm_chain.predict()
        chat_history += [AIMessage(content=ai_output)]

        plan = self.parse_plan(ai_output)
        logger.info(plan)

        return ExecutionOutput(
            state_update={"plan": plan},
            chat_history=chat_history,
        )

    def handle_observation(self, observation, chat_history, **kwargs):
        msg = observation.get_message(suffix= \
            "\nTo systematically fix this issue, answer the following questions one by one:\n"
            "1. What are potential causes for this error? Think step by step.\n"
            "2. Explain in detail how this error could be fixed.\n"
            "3. Are there additional relevant columns necessary to fix the error(Yes / No)?\n"
        )
        prompt = ChatPromptTemplate.from_messages(chat_history + [msg])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answers = chain.predict()
        logger.warning(observation)
        logger.warning(answers)
        _, fix_idea, additional_cols_str = \
            [x.strip() for x in re.split(r"(^1\.|\n2\.|\n3\.)", answers)[slice(2, None, 2)]]
        additional_cols = "yes" in re.split(r"\W", additional_cols_str.lower())
        if additional_cols:
            which_info = re.search(r"(^|\W)((Y|y)es)\W+(\w.+)\W", additional_cols_str)[4].strip(" .")
            raise ExecutionError(description=f"This information is missing: {which_info}",
                                 target_phase=DiscoveryPhase)
        observation.set_fix_idea(fix_idea)

        chat_history.append(observation.get_message(suffix="\nPlease come up with a fixed plan."))
        return chat_history

    def reinit_chat(self, observation, chat_history, query, relevant_columns, **kwargs):
        chat_history.append(observation.get_message(suffix="\nPlease come up with a fixed plan."))
        chat_history.append(HumanMessagePromptTemplate.from_template(REINIT_PROMPT))
        chat_history = ChatPromptTemplate.from_messages(chat_history).format_prompt(
            relevant_columns=relevant_columns.with_join_hints()
        ).messages
        return chat_history

    def parse_plan(self, plan):
        available_tables = set(self.database.tables.keys())
        result = []
        plan = plan.split("\n")
        current_step = None
        for step in plan:
            if step.startswith("Step"):
                if current_step is not None:
                    result.append(current_step)
                step_str = ":".join(step.split(":")[1:]).strip()
                current_step = PlanStep(step_str, available_tables=available_tables)
            if step.startswith("Input"):
                step_str = ":".join(step.split(":")[1:]).strip()
                current_step.set_input(step_str.split(","))
            if step.startswith("Output"):
                step_str = ":".join(step.split(":")[1:]).strip()
                if step_str != "N/A":
                    current_step.set_output(step_str)
                    available_tables.add(step_str)
            if step.startswith("New Columns"):
                step_str = ":".join(step.split(":")[1:]).strip()
                if step_str != "N/A":
                    current_step.set_new_columns(step_str.split(","))
        if current_step is not None and current_step.input_tables != []:
            result.append(current_step)
        result = self.filter_steps(result)
        return Plan(result)

    def filter_steps(self, steps):
        forbidden = {"verify", "make sure", "confirm", "test", "validate", "if", "in case", "double-check", "check"}
        steps = [
            s for s in steps
            if not any(s.description.lower().startswith(prefix) for prefix in forbidden) and len(s.input_tables) > 0
        ]
        return steps


FORMAT_INSTRUCTIONS = """
Use the following format:
Request: The user request you must satisfy by using your capabilities
Thought: You should always think what to do.
Step 1: Description of the step.
Input: List of tables passed as input. Usually this is a single table, except when tables need to be combined using a join.
Output: Name of the output table. N/A if there is no output e.g. when plotting.
New Columns: The new columns that have been added to the dataset. N/A if no column has been added.
... (this Step/Input/Output/New Columns can repeat N times)
Step N: Plan completed.
"""