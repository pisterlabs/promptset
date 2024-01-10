"""Inner Dialog
Input: A open-ended question (things to consider before starting a business).
Output: A text file containing the final result.
"""
import openai
import google.generativeai
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import os, json, re 
from dotenv import load_dotenv
from inner_dialog.gemini import article2hiercc_gemini


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

max_supervisor_turn = 7
interview_turn = 3
end_token = "<END>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="inner_dialog.log",
    filemode="w",
)


class Agent:
    def __init__(self, model: ChatGoogleGenerativeAI, system_message: str) -> None:
        self.model = model
        self.system_message = system_message
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message


class Supervisor(Agent):
    gen_subtopics_template = HumanMessagePromptTemplate.from_template(
        template="""
        give me {num_subtopics}  sub-topic that we can discuss about 
        the question: {question}
        Output the sub-topics (Cannot contain numbers, only text) in JSON format with key "subtopics".
        Output format:
        {{
            "subtopics": [
                " ",
            ]
        }}
        """
    )
    extract_keypoints_template = HumanMessagePromptTemplate.from_template(
        template="""
        Topic: {topic}
        Dialogue:
        {dialog}

        Can you extract 5 key point (only Verb+Object pairs) that the expert just answer? 
        Keep the note bulletlisted and concise without sacrificing any important information.
        Example output format :
        ```
        * " ": " "
        * " ": " "
        ```
        """
    )
    integrate_note_template = HumanMessagePromptTemplate.from_template(
        template="""
        Note: {note}
        Keypoints:
        {keypoints}

        Integrate the keypoints you have extracted to the notes. All notes must be in the same format. Use bullet points.
        Example output format :
        ```
        * sub-topic1
            * " ": " "
            * " ": " "
        ```
        """
    )

    def __init__(
        self,
        question: str,
        model: ChatGoogleGenerativeAI,
        system_message: SystemMessage,
        num_subtopics: int = 10,
    ):
        super().__init__(model=model, system_message=system_message)
        self.note = ""
        self.question = question
        self.subtopics = []
        self.num_subtopics = num_subtopics
        self.current_subtopic = ""

    def _parse_subtopics(self, subtopics_str: str) -> list[str]:
        """Expecting format:
        {
            "subtopics": [
                "Risk assessment and volatility",
                "Researching and understanding different cryptocurrencies",
                "Security measures and safeguards"
            ]
        }
        """
        print("subtopics_str:", subtopics_str)
        print("="*100)
        json_data = re.search(r'{\n[^`]+}', subtopics_str).group(0)
        print("json_data:", json_data)
        st_json = json.loads(json_data)
        return st_json["subtopics"]

    def regen_subtopics(self):
        """Clear subtopic queue and re-generate subtopic
        related to original question.

        The subtopic should include relevant context (or the
        original question) because asker doesn't know what
        the original question is.

        When generating a subtopic, we should also consider
        the notes cause we don't want to generate subtopic
        we already discuss.

        Append `end_token` at the end to mark end of discussion
        when all the subtopics are being discussed.
        """
        human_msg = self.gen_subtopics_template.format_messages(
            question=self.question,
            num_subtopics=self.num_subtopics,
            # TODO: should remember previous disscuss sub topic or notes.
        )[0]
        ai_msg = self.step(human_msg)
        self.subtopics = self._parse_subtopics(ai_msg.content)
        self.subtopics.append(end_token)
        logging.info(f"Supervisor re-gen subtopics: {self.subtopics}")

    def next_subtopic(self, regen: bool = False):
        if len(self.subtopics) == 0 or regen:
            self.regen_subtopics()
        self.current_subtopic = self.subtopics.pop(0)

    def conclude(self, dialog: list[str]):
        try:
            keypoints = self.extract_keypoints(dialog)
            self.integrate(keypoints)
            # NOTE:
            # To avoid exceeding 4097 token limit.
            # Supervisor only need to remember
            # the subtopic it has discuss and the note.
            self.reset()
        except openai.error.InvalidRequestError as e:
            logging.error(f"open ai invalid request error: {e}")
            logging.error(f"Stored supervisor message: {self.stored_messages}")

    def _dialog_to_str(self, dialog: list[str]) -> str:
        output = ""

        i = 0
        while i < len(dialog):
            output += f"""Asker: {dialog[i]}
            Answerer: {dialog[i+1]}
            """
            i += 2

        return output

    def extract_keypoints(self, dialog: list[str]) -> str:
        """Extract keypoint based on dialog and sub topic.
        Return keypoints: str
        """
        human_msg = self.extract_keypoints_template.format_messages(
            topic=self.current_subtopic,
            dialog=self._dialog_to_str(dialog),
        )[0]
        ai_msg = self.step(human_msg)
        logging.info(f"Supervisor extracted keypoints: {ai_msg.content}")
        return ai_msg.content

    def _llm_integrate(self, keypoints: str) -> None:
        """Prompt LLM to integrate keypoints and note."""
        human_msg = self.integrate_note_template.format_messages(
            note=self.note,
            keypoints=keypoints,
        )[0]
        ai_msg = self.step(human_msg)
        self.note = ai_msg.content

    def _naive_integrate(self, keypoints: str) -> None:
        """Directly append keypoints to note."""
        self.note += f"\n{keypoints}"

    def integrate(self, keypoints: str) -> None:
        """Integrate keypoints and note to form new notes.
        Use self.note and keypoints
        Modify self.note
        """
        # self._llm_integrate(keypoints)
        self._naive_integrate(keypoints)
        logging.info(f"Supervisor integrate: {self.note}")


class Asker(Agent):
    init_subq_template = HumanMessagePromptTemplate.from_template(
        template="Ask one question to the expert considering {topic} in turns of {question}."
    )
    next_subq_template = HumanMessagePromptTemplate.from_template(
        template="""Here is the reply:
        {subanswer}
        
        Ask the 1 most critical questions to drive deeper discussions.
        """
    )

    def gen_init_subquestion(self, topic: str, question: str) -> str:
        """Generate initial sub question."""
        human_msg = self.init_subq_template.format_messages(
            topic=topic,
            question=question,
        )[0]
        ai_msg = self.step(human_msg)
        return ai_msg.content

    def gen_next_subquestion(self, subanswer: str) -> str:
        """Geneate next sub questions."""
        human_msg = self.next_subq_template.format_messages(subanswer=subanswer)[0]
        ai_msg = self.step(human_msg)
        return ai_msg.content


class Answerer(Agent):
    def gen_subanswer(self, subquestion: str) -> str:
        """Generate sub answer for the subquestion."""
        human_msg = HumanMessage(content=subquestion)
        ai_msg = self.step(human_msg)
        return ai_msg.content


def t2cb_ask_inner_dialog(question: str) -> str:
    """Ask T2CB question to inner dialog.

    Args:
        question (str): Things to consider before ...

    Returns
        str: Inner dialog's response.
    """
    asker = Asker(
        system_message=SystemMessage(
            content="""
            You ask a critical questions that
            encourage deeper thinking, analysis, 
            and evaluation of a given topic or situation. 
            Critical questions that are designed to challenge assumptions, 
            uncover biases, examine evidence, and promote 
            a more thorough understanding of complex issues.
            In addition, restrict to choose 1 critical questions to drive deeper discussions.
            """
        ),
        model=ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True),
    )
    answerer = Answerer(
        system_message=SystemMessage(
            content="""
            You give short but concise reply (only Verb+Object pairs) to each question.
            """,
        ),
        model=ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True),
    )
    supervisor = Supervisor(
        question=question,
        # TODO: Maybe a better system prompt than this?
        system_message=SystemMessage(
            content="""
            You are a experience supervisor. 
            You must extract the key points of all the dialogues in each round and present them in the form of bullet points, 
            and integrate the key points taken out in all rounds and also present them in the form of bullet points. 
            """,
        ),
        model=ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True),
    )

    # Reset
    asker.reset()
    answerer.reset()

    dialog = list()
    logging.info(f"Original Question: {question}")
    for _ in range(max_supervisor_turn):
        supervisor.next_subtopic()
        if supervisor.current_subtopic == end_token:
            break
        logging.info(f"Supervisor: Topic is {supervisor.current_subtopic}")
        for i in range(interview_turn):
            if i == 0:
                sub_question = asker.gen_init_subquestion(
                    supervisor.current_subtopic,
                    question,
                )
            else:
                sub_question = asker.gen_next_subquestion(sub_answer)
            dialog.append(sub_question)
            logging.info(f"Asker: {sub_question}")
            sub_answer = answerer.gen_subanswer(sub_question)
            dialog.append(sub_answer)
            logging.info(f"Answer: {sub_answer}")
        supervisor.conclude(dialog)
        # Clear dialog. (gpt-3.5-turbo max token 4097)
        dialog = []
        # NOTE: Clear dialog history of asker and answerer otherwise it
        # will exceed token limit. (Event for chatmodel?)
        #
        # Not sure will they repeat the same topic because they
        # forgot what they have talked about before?
        asker.reset()
        answerer.reset()

    return supervisor.note


if __name__ == "__main__":
    # # question = "how to practice self-love and gain confidence?"
    # # question = "things to consider before traveling to a foreign country"
    question = "things to consider before starting a podcast"
    article = t2cb_ask_inner_dialog(question=question)
    hierr = article2hiercc_gemini(article, question=question)
    logging.info(f"article2hierr: \n{json.dumps(hierr, indent=2)}")
    with open("inner_dialog_output", "w") as f:
        f.write(article)
