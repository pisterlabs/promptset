import openai
from storyconnect.settings import OPENAI_API_KEY
from .exceptions import ContinuityCheckerNullTextError
from .models import StatementSheet

# from .models import StatementSheet
import lxml.etree as etree
import re
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


class ContinuityChecker:
    # openai parameters
    BASE_MODEL = "gpt-3.5-turbo-instruct"
    CHAT_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.2

    # # continuity statement parameters
    # stmt_task = "Generate a number of simple statements about each named entity in the text that are directly supported by the text. The statements only describe the named entities and do not describe actions within the text."
    # stmt_instructions = "Format the response in XML, the first chunk will be for characters and the second will be for places. The header for each subsection is the entity's name, and what follows are the statements about the entities. The statements must start with the entity's name and not a pronoun. If the entity is more than one word, replace the space with a hyphen in the element tag."
    # stmt_ex = "<Characters>\n<John-Doe>\nJohn Doe has blue eyes.\nJohn Doe is tall.\n</John-Doe>\n<Jane-Doe>\nJane Doe has blonde hair.\nJane Doe is 23 years old.\n</Jane-Doe>\n</Characters>\n<Locations>\n<New-York>\nNew York is a city.\nNew York is in the United States.\n</New-York>\n\n</Locations>"

    # continuity comparison parameters

    last_response = None

    @staticmethod
    def _strip_xml(text):
        tree = etree.fromstring(text)

        statements = ""
        for child in tree:
            for subchild in child:
                statements += subchild.text.strip()

        return statements

    @staticmethod
    def _statement_prompt(text):
        # continuity statement parameters
        stmt_task = "Generate a number of simple statements about each named entity in the text that are directly supported by the text. The statements only describe the named entities and do not describe actions within the text."
        stmt_instructions = "Format the response in XML, the first chunk will be for characters and the second will be for places. The header for each subsection is the entity's name, and what follows are the statements about the entities. The statements must start with the entity's name and not a pronoun. If the entity is more than one word, replace the space with a hyphen in the element tag."
        stmt_ex = "<Characters>\n<John-Doe>\nJohn Doe has blue eyes.\nJohn Doe is tall.\n</John-Doe>\n<Jane-Doe>\nJane Doe has blonde hair.\nJane Doe is 23 years old.\n</Jane-Doe>\n</Characters>\n<Locations>\n<New-York>\nNew York is a city.\nNew York is in the United States.\n</New-York>\n\n</Locations>"

        return f"Text: {text}\n\nTask: {stmt_task}\n\nInstructions: {stmt_instructions}\n\n Example: \n{stmt_ex}"

    def create_statementsheet(self, text):
        if text == "":
            raise ContinuityCheckerNullTextError()

        prompt = ContinuityChecker._statement_prompt(text)
        self.last_response = client.completions.create(
            model=self.BASE_MODEL,
            prompt=prompt,
            # max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            requst_timeout=300,
        )

        body = self.last_response.choices[0].text

        formatted_text = "<Statements>\n" + body.strip() + "\n</Statements>"
        return formatted_text

    def filter_statementsheet(self, statementsheet):
        filter_prompt = """Given the following set of statements, remove 
        any statements that do not define the entities they are describing. Remove any statemnents that 
        describe what is currently happening. 
        Example: Given "John Doe is tall." and "John Doe is running." remove "John Doe is running."\n"""
        filter_prompt += statementsheet

        self.last_response = client.completions.create(
            model=self.BASE_MODEL,
            prompt=filter_prompt,
            # max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            timeout=300,
        )
        return self.last_response.choices[0].text.strip()

    def compare_statementsheets(self, s_old, s_new):
        s_old = ContinuityChecker._strip_xml(s_old)
        s_new = ContinuityChecker._strip_xml(s_new)

        # print(s_old)
        # print('\n')
        # print(s_new)
        comp_input = f"The following statements are about previously written text: \n {s_old} \n The next statements are about new text: \n {s_new}\n"
        # TODO: unused variable
        # comp_instructions = "Identify any contradictions between the old text and the new text. Briefly summarize the contradicitons. Do not say anything about changes that dont contain contradictions. Do not use complicated formatting. Be brief. Stop when appropriate."
        comp_instructions_list = "List any contradictions between the descriptions in the old text versus the new. If an entity is not mentioned in the new text, ignore it. If no contradicions exist, say 'NONE'."
        prompt = comp_input + comp_instructions_list

        self.last_response = client.completions.create(
            model=self.BASE_MODEL,
            prompt=prompt,
            # max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE * 2,
            timeout=300,
        )

        response = self.last_response.choices[0].text.strip()
        pattern = re.compile(r"(\d+\. )")
        return re.sub(pattern, "", response)


class ContinuityCheckerChat:
    # openai parameters
    BASE_MODEL = "gpt-3.5-turbo-instruct"
    CHAT_MODEL = "gpt-3.5-turbo-1106"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.2

    # # continuity statement parameters
    # ...

    # continuity comparison parameters

    last_response = None

    @staticmethod
    def _strip_xml(text):
        tree = etree.fromstring(text)

        statements = ""
        for child in tree:
            for subchild in child:
                statements += subchild.text.strip()

        return statements

    @staticmethod
    def _statement_prompt(text):
        # continuity statement parameters
        stmt_task = "Generate a number of simple statements about each named entity in the text that are directly supported by the text. The statements only describe the named entities and do not describe actions within the text."
        stmt_instructions = "Format the response in XML, the first chunk will be for characters and the second will be for places. The header for each subsection is the entity's name, and what follows are the statements about the entities. The statements must start with the entity's name and not a pronoun. If the entity is more than one word, replace the space with a hyphen in the element tag."
        stmt_ex = "<Characters>\n<John-Doe>\nJohn Doe has blue eyes.\nJohn Doe is tall.\n</John-Doe>\n<Jane-Doe>\nJane Doe has blonde hair.\nJane Doe is 23 years old.\n</Jane-Doe>\n</Characters>\n<Locations>\n<New-York>\nNew York is a city.\nNew York is in the United States.\n</New-York>\n\n</Locations>"

        return f"Text: {text}\n\nTask: {stmt_task}\n\nInstructions: {stmt_instructions}\n\n Example: \n{stmt_ex}"

    def create_statementsheet(self, text):
        if text == "":
            raise ContinuityCheckerNullTextError()

        prompt = ContinuityChecker._statement_prompt(text)
        client = OpenAI(api_key=OPENAI_API_KEY)
        self.last_response = client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            timeout=300,
        )

        body = self.last_response.choices[0].message.content
        with open("ai_features/test_files/test_prints.txt", "w") as f:
            f.write(str(self.last_response))

        formatted_text = "<Statements>\n" + body.strip() + "\n</Statements>"
        return formatted_text

    def filter_statementsheet(self, statementsheet):
        filter_prompt = """Given the following set of statements, remove 
        any statements that do not define the entities they are describing. Remove any statemnents that 
        describe what is currently happening. 
        Example: Given "John Doe is tall." and "John Doe is running." remove "John Doe is running."\n"""
        filter_prompt += statementsheet
        client = OpenAI(api_key=OPENAI_API_KEY)

        self.last_response = client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[{"role": "system", "content": filter_prompt}],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            timeout=300,
        )
        return self.last_response.choices[0].message.content.strip()

    def compare_statementsheets(self, s_old, s_new):
        s_old = ContinuityChecker._strip_xml(s_old)
        s_new = ContinuityChecker._strip_xml(s_new)

        # print(s_old)
        # print('\n')
        # print(s_new)
        comp_input = f"The following statements are about previously written text: \n {s_old} \n The next statements are about new text: \n {s_new}\n"
        comp_instructions = "Identify any contradictions between the old text and the new text.  Do not say anything about changes that dont contain contradictions. Do not use complicated formatting. Be brief. Stop when appropriate."
        comp_instructions_list = "List any contradictions between the descriptions in the old text versus the new. If an entity is not mentioned in the new text, ignore it. If no contradicions exist, say 'NONE'."
        prompt = comp_input + comp_instructions_list
        client = OpenAI(api_key=OPENAI_API_KEY)

        self.last_response = client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=self.TEMPERATURE * 2,
            max_tokens=self.MAX_TOKENS,
            timeout=300,
        )

        response = self.last_response.choices[0].message.content.strip()
        pattern = re.compile(r"(\d+\. )")
        return re.sub(pattern, "", response)
    


