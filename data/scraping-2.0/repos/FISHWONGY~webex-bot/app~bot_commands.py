from webex_bot.models.command import Command
from common_func import *
from utils import jira_project_key
import re
import openaiapi as openaiapi
import jiraapi as jiraapi


jira_lookup_dict = get_jira_user_id()


class SqlDebug(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!sqldebug",
            help_message=f"Unrecognised command.\n"
            f"Please type !sqldebug followed by your snowflake query then #error with your error message.",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        try:
            q_str, err_str = re.split(
                pattern="#error", string=message, maxsplit=1, flags=re.IGNORECASE
            )
            if not err_str.strip():
                raise ValueError("Error message not provided after '#error'")
            msg_hist = openaiapi.construct_sf_prompt(q_str, error=err_str)
            response_message = openaiapi.chat_response(msg_hist)
        except ValueError:
            response_message = (
                "Invalid input format. \nPlease type !sqldebug followed by your snowflake query then #error with your error message. "
                "\nExample: \n\n```\n!sqldebug SELECT col FROM tbl\n\n#error SQL compilation error: error line 1 at position 7 invalid identifier 'col'\n```"
            )

        return response_message


class SqlOpt(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!sqlopt",
            help_message=f"Unrecognised command.\n"
            f"Please type !sqlopt followed by your snowflake query then #opt with your optimisation related message.",
            card=None,
        )

    def pre_execute(self, message, attachment_actions, activity):
        return "<blockquote class=info>\n\n**Command received.**\n\n\n**Working on SQL Optimisation...**\n\n</blockquote>"

    def execute(self, message, attachment_actions, activity):
        try:
            q_str, comm_str = re.split(
                pattern="#opt", string=message, maxsplit=1, flags=re.IGNORECASE
            )
        except ValueError:
            q_str = message
            comm_str = None

        q_str = q_str.replace("!sqlopt", "").strip()

        if q_str == "":
            response_message = (
                "Invalid input format. \nPlease type !sqlopt followed by your snowflake query. Optionally, you can also add #opt with your optimisation related message. "
                "\nExample: \n\n```\n!sqlopt SELECT col FROM tbl\n\n#opt There is a date filter called 'update_date'\n```"
                "\nor\n\n"
                "\nExample: \n\n```\n!sqlopt SELECT col FROM tbl\n\n```"
            )
        else:
            msg_hist = openaiapi.construct_sf_prompt(q_str, comment=comm_str)
            response_message = openaiapi.chat_response(msg_hist)

        return response_message


class SqlFormatter(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!sqlformat",
            help_message=f"Unrecognised command.\n"
            f"Please type !sqlformat followed by your snowflake query",
            card=None,
        )

    def pre_execute(self, message, attachment_actions, activity):
        return "<blockquote class=info>\n\n**Command received.**\n\n\n**Working on SQL Reformatting...**\n\n</blockquote>"

    def execute(self, message, attachment_actions, activity):
        q_str = message.strip().lower()

        if q_str == "":
            response_message = (
                "Invalid input format. \nPlease type !sqlformat followed by your snowflake query."
                "\nExample: \n\n```\n!sqlformat SELECT col1, col2, col3 FROM tbl\n\n```"
            )
        else:
            msg_hist = openaiapi.construct_formatter_prompt(query=q_str)
            response_message = openaiapi.chat_response(msg_hist)

        return response_message


class ChatAI(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!ai",
            help_message="Unrecognised command.\nPlease type !ai to get Chat GPT link",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        message = message.strip().lower()

        if message == "":
            response_message = "[Chat GPT](https://chat.openai.com)"
        else:
            response_message = (
                "Unrecognised command.\nPlease type !ai to get Chat GPT link"
            )

        return response_message


class CodeHelp(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!codeq",
            help_message="Unrecognised command.\nPlease type !codeq followed by -l {lang} and -q {question}.",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        try:
            lang = extract_value(r"-l\s*([\w\s]+)", message)
            question = extract_value(r"-q\s*(.+)", message)

            if not lang or not question:
                raise ValueError

            msg_hist = openaiapi.construct_code_help_prompt(lang, question)
            response_message = openaiapi.chat_response(msg_hist)
        except ValueError:
            response_message = (
                "Invalid input format. \nPlease type !codeq followed by -l {lang} with the programming language you are enquiring and -q {question} with your question."
                "\nExample: \n\n```\n!codeq -l python -q How can I print hello world?\n```"
                "\nOr"
                "\n```\n!codeq -q How can I print hello world? -l python\n```"
            )

        return response_message


class JiraStory(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!jsget",
            help_message="Unrecognised command.\nPlease type !jsget followed by #{role} and your jira story title.",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        try:
            role, title = message[1:].split(" ", 1)
            role = role.lower()
            title = title[0].upper() + title[1:]

            if role == "#de":
                role = "data engineer"
            elif role == "#da":
                role = "data analyst"
            else:
                raise ValueError

            msg_hist = openaiapi.construct_jira_prompt(role, title)
            response_message = openaiapi.chat_response(msg_hist)
        except ValueError:
            response_message = (
                "Invalid input format. \nPlease type !jsget followed by #{role: da/ de} and your jira story title."
                "\nExample: \n\n```\n!jsget #de Create new data pipeline\n```"
            )

        return response_message


class JiraStoryWrite(Command):
    def __init__(self):
        super().__init__(
            command_keyword="!jspost",
            help_message="Unrecognised command.\nPlease type !jspost followed by #{role} and your jira story title.",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        try:
            role, remaining_msg = message[1:].split(" ", 1)
            role = jiraapi.parse_role(role)

            username = extract_value(r"-u\s*(\w*)", remaining_msg)
            epic_id = extract_value(r"-e\s*(\w*)", remaining_msg)
            team_id = extract_value(r"-t\s*(\w*)", remaining_msg)
            sp = extract_value(r"-sp\s*(\w*)", remaining_msg)

            remaining_msg = re.sub(
                r"\s*-u\s*\w+|\s*-e\s*\w+|\s*-t\s*\w+|\s*-sp\s*\w+", "", remaining_msg
            ).strip()
            title = remaining_msg[0].upper() + remaining_msg[1:]

            sender_email = activity["actor"]["emailAddress"]
            (
                default_team_id,
                default_jira_id,
                default_epic_id,
            ) = get_user_deatil_from_email(jira_lookup_dict, sender_email)

            if username:
                user_email = f"{username.lower()}@email.com"
                if user_email not in jira_lookup_dict:
                    raise ValueError("Invalid user id.")
                _, default_jira_id, _ = get_user_deatil_from_email(
                    jira_lookup_dict, user_email
                )
            if epic_id:
                default_epic_id = f"{jira_project_key}-{epic_id}"
            if team_id:
                if team_id not in ["123", "456"]:
                    raise ValueError("Invalid team id.")
                default_team_id = team_id

            if username is None:
                assignee = sender_email.split("@")[0].lower()
            else:
                assignee = username

            msg_hist = openaiapi.construct_jira_prompt(role, title)
            response_message = openaiapi.chat_response(msg_hist)
            logger.info(f"{response_message}")
            response_dict = extract_json_string(response_message)
            jiraapi.write_jira_story(
                jiraapi.get_jira_story_details(
                    response_dict, default_team_id, default_jira_id, default_epic_id, sp
                )
            )

            

            response_message = f"{response_message}\n{sender_email} Created a jira story for {assignee.lower()} under epic {default_epic_id} on {[Jira Board](https://google.com)}"

        except ValueError as e:
            response_message = str(e).capitalize() + (
                "\n\nInvalid input format. \nPlease type !jspost followed by #{role: da/de} and your jira story title.\nOptionally you can have -u {user_cec}, -e {epic-id} and -t {jira-board-id} as well."
                "\nExample: \n\n```\n!jspost #de Create new data pipeline\n```"
                "\n\nOptionally, you could also define your own parameter for **user**, **epic** and **team id** as below:"
                "\n\nDefining user and epic:"
                "\n```\n!jspost #de Create new data pipeline -u 123 -e 456\n```"
                "\nDefining user only"
                "\n```\n!jspost #de Create new data pipeline -u 123 \n```"
                "\nDefining epic only:"
                "\n```\n!jspost #de Create new data pipeline -e 456\n```"
            )
        except Exception as e:
            response_message = f"Failed to create a jira story: {str(e)}"

        return response_message
