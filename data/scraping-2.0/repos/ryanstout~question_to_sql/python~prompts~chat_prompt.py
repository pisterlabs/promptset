import datetime
from pprint import pprint

from python.prompts.prompt import Prompt
from python.schema.ranker import SCHEMA_RANKING_TYPE, merge_schema_rankings
from python.schema.schema_builder import SchemaBuilder
from python.sql.sql_parser import SqlParser
from python.utils.db import application_database_connection
from python.utils.logging import log
from python.utils.openai import openai_engine
from python.utils.tokens import count_tokens

db = application_database_connection()


class ChatPrompt(Prompt):
    section_order: list[str] = ["prologue", "few_shot", "question"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_global_instruct = True
        self.backticks = "```"
        self.comments = ""
        self.few_shot_in_system = True
        self.rules_after_schema = True

    def available_tokens(self) -> int:
        # extra buffer, since SchemaBuilder doesn't match exactly
        extra_buffer_for_schema_builder = 100
        completion_tokens = 1_024

        engine = openai_engine()

        match engine:
            case "gpt-4":
                start_tokens = 8_000
            case "gpt-4-32k":
                start_tokens = 32_000
            case _:
                start_tokens = 4_096
        return start_tokens - completion_tokens - extra_buffer_for_schema_builder

    def generate(self) -> list[dict]:
        sections = {}
        few_shot_text = ""
        rules = ""
        if "prologue" in self.section_order:
            prologue = self.prologue()
            if len(prologue) > 0:
                prologue = prologue[0]["content"] + "\n\n"
            else:
                prologue = ""

            if self.use_rules:
                rules = self.rules()[0]["content"]

            if self.few_shot_generator and self.few_shot_in_system and "few_shot" in self.section_order:
                few_shot_data, few_shot_rankings = self.few_shot(self.user_question)

                few_shot_lines: list[str] = []
                for few_shot in few_shot_data:
                    if few_shot["role"] == "assistant":
                        few_shot_lines.append(few_shot["content"])
                        few_shot_lines.append("\n")
                    else:
                        few_shot_lines.append(f"Question: {few_shot['content']}")

                few_shot_text = "\n".join(few_shot_lines)
                few_shot_text = f"\n\nHere's a few examples:\n\n{few_shot_text}\n\n"

                # Merge the few_shot_rankings into the schema rankings
                self.ranked_schema = merge_schema_rankings(self.ranked_schema, few_shot_rankings)

            role = "system"
            if not self.rules_after_schema:
                prologue += rules

            sections["prologue"] = [{"role": role, "content": prologue}]

        if (not self.few_shot_in_system) and "few_shot" in self.section_order:
            sections["few_shot"], few_shot_rankings = self.few_shot(self.user_question)

            # Merge the few_shot_rankings into the schema rankings
            self.ranked_schema = merge_schema_rankings(self.ranked_schema, few_shot_rankings)

        if "question" in self.section_order:
            sections["question"] = self.question()

        # Count the used tokens so far
        self.used_tokens = 0
        for section in sections.values():
            contents = map(lambda x: x["content"], section)
            self.used_tokens += sum(map(count_tokens, contents))

        if self.rules_after_schema:
            self.used_tokens += count_tokens(rules)

        # Compute the schema using the remaining tokens
        sections["prologue"][0]["content"] += "\n\n"
        sections["prologue"][0]["content"] += self.schema(self.available_tokens() - self.used_tokens)[0]["content"]

        if self.rules_after_schema:
            sections["prologue"][0]["content"] += f"\n\n{rules}"

        # sections["prologue"][0][
        #     "content"
        # ] += "\n\nWhen generating a SQL Query, any columns must be on the correct table in the provided Schema!"

        if self.few_shot_generator and self.few_shot_in_system and "few_shot" in self.section_order:
            sections["prologue"][0]["content"] += f"\n{few_shot_text}"

        prompt = []

        # Add the sections in the order specified by the section_order attribute
        for section in self.section_order:
            if self.few_shot_in_system and section == "few_shot":
                continue
            prompt += sections[section]

        for line in prompt:
            log.debug("prompt: ", role=line["role"], content=line["content"])

        return prompt

    def prologue(self):
        if self.use_global_instruct:
            return [
                {
                    "role": "system",
                    # "content": "You are an expert SQL developer. Given a Schema, and a Question, generate the correct SQL Query that answers the users question using the Schema. Only return the SQL Query that answers the question and no other text. The SQL Query must follow the Rules and Schema provided below.",
                    # "content": "You are an expert SQL developer. Given the following Schema and a Question, generate the correct SQL Query that answers the Question using the tables and columns in the Schema. Only return the SQL Query that answers the question and no other text. The SQL Query should also follow the Rules provided after the Schema below. All tables and columns must be described in the Schema to be used in the SQL Query.",
                    # "content": "Use the following sql Schema to return the SQL query that correctly answers the provided questions. Only return the SQL Query and no other text.",
                    # "content": "I want you to act as a SQL expert. The database you will use is described by the following schema. I will type questions and you will reply with the correct SQL to answer the question. Do not write explanations. Make sure you generate SQL Queries that match the schema.",
                    "content": "You are an expert SQL developer. Given the following Schema and Rules, generate the correct SQL Query that answers the users question. Return only the SQL Query that answers the Question, follows the Rules, and uses only tables/columns from the Schema in a code block.",
                }
            ]
        else:
            return []

    def schema(self, available_tokens: int):
        """
        Returns a messsage represenation of the schema with associated prologue
        """
        schema = SchemaBuilder().build(self.data_source_id, self.ranked_schema, available_tokens)

        return [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "Schema:",
                        f"```\n-- {SqlParser.in_dialect.capitalize()} SQL schema\n{schema}\n```",
                    ]
                ),
            }
        ]

    def rules(self):
        """
        Rules we provide to the LLM to attempt to get it to do what we want. This will probably work better on chatgpt
        than others, but seems to work in a lot of cases on codex.
        """

        def generate_date_suffix(n):
            if 10 <= n % 100 < 20:
                return "th"

            return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

        day_int = datetime.datetime.now().day
        suffix = generate_date_suffix(day_int)

        # current date in the format of 'January 28th, 2023'
        # the date format here is picked in a very specific fashion
        current_date = datetime.datetime.now().strftime(f"%B %-d{suffix}, %Y")

        if self.use_rules:
            rule_num: int = 0

            def rule_prefix():
                return ""
                nonlocal rule_num
                rule_num += 1
                return f"{rule_num}. "

            if SqlParser.in_dialect == "postgres":
                now_function = "NOW()"
            else:
                now_function = "CURRENT_TIMESTAMP()"

            rules = [
                "",
                # f"{comments}Rules for building SQL queries: ",
                "Rules: ",
                # f"{rule_prefix()}Return `SELECT 'unsure';` if the SQL for the question can not be generated",
                # f"{rule_prefix()}Do case insensitive matches using LOWER unless the case matters or it matches a possible value",
                f"{rule_prefix()}When matching a string, use LOWER or ILIKE unless it matches a listed possible value.",
                # f"{rule_prefix()}Calculate lifetime of a customer by taking the duration between the first and most recent order for a customer. ",
                f"{rule_prefix()}If we're returning a day, always also return the month and year.",
                f"{rule_prefix()}Add ORDER BY to every SELECT.",  # makes things deterministic
                # In snowflake, Count(*), etc.. are not allowed directly in a ORDER BY or HAVING, they have to be in the
                # SELECT expression first, then accessed by alias.
                f"{rule_prefix()}COUNT's used in ORDER BY or HAVING should appear in the SELECT first.",
                # f"{rule_prefix()}Any columns used must be in the Schema",
                f"{rule_prefix()}Assume the current date is {current_date}.",
                f"{rule_prefix()}Use {now_function} instead of dates.",
                f"{rule_prefix()}Don't use CTE's.",
                f"{rule_prefix()}The SQL Query should start with `SELECT` and not `WITH`.",
                f"{rule_prefix()}The SQL Query should be in {SqlParser.in_dialect.capitalize()} dialect.",
                f"{rule_prefix()}Qualify all columns with the table name.",  # prevents ambigious columns issue
                # f"{rule_prefix()}Look at the Schema to decide what tables and columns to use. (Don't use any that aren't in the Schema)",
                # f"{rule_prefix()}When generating a SQL Query, any columns must be on the correct table in the provided Schema!",
                # f"{rule_prefix()}All columns must be on its associated table in the Schema!",
                # f"{rule_prefix()}Always use table_name.column_name instead of leaving columns ambiguous",
                # f"{rule_prefix()}Don't alias tables unless necessary",
                # f"{rule_prefix()}GROUP BY clauses should include all fields from the SELECT clause",
                "",
            ]
        else:
            rules = []

        return [
            {
                "role": "system",
                "content": "\n".join(rules),
            }
        ]

    def few_shot(self, current_question: str) -> tuple[list[dict[str, str]], SCHEMA_RANKING_TYPE]:
        # The generator will only be present if we've enabled adding few shot
        if self.few_shot_generator:
            few_shot_examples, correct_sql, schema_rankings = self.few_shot_generator.generate(current_question)

            examples: list[dict[str, str]] = []

            for idx, example in enumerate(few_shot_examples):
                examples.append(
                    {
                        "role": "user",
                        "content": example,
                    }
                )
                examples.append({"role": "assistant", "content": f"```{correct_sql[idx]}```"})

            return (examples, schema_rankings)
        else:
            return ([], [])

    def question(self):
        return [{"role": "user", "content": self.user_question}]

    def is_chat(self):
        return True
