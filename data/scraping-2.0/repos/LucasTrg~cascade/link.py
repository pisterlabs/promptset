import logging
from typing import List

logging.basicConfig(level=logging.DEBUG)

import jinja2
import openai

import constraint
import conversation


class Link:
    def __init__(
        self,
        prompt_template: str,
        constraints: List[constraint.Constraint],
        executor,
        *args,
        **kwargs,
    ):
        self.prompt_template = prompt_template
        self.constraints = constraints
        self.executor = executor

    def execute(
        self,
        conversation: conversation.Conversation,
        repeat_threshold: int = 5,
        best_effort: bool = False,
    ) -> conversation.Conversation:
        """Takes a list of past messages and returns

        Args:
            past_messages (Conversation): List of past messages
            repeat_threshold (int, optional): Maximum number of times the link should be replayed in case of error. Defaults to 5.
            best_effort (bool, optional): If False, the link will raise an error and interrupt the chain if it fails to generate a valid response. If True, the link will return the last response it could generate. Defaults to False.


        Returns:
            Conversation : Conversation with the link prompt and result appended.
        """

        result = " "
        prompt = self.build_prompt(conversation)
        conversation.add_message({"role": "user", "content": prompt, "keep": True})
        logging.debug("Conversation" + str(conversation.messages))
        for i in range(repeat_threshold):
            logging.debug("Attempt" + str(i))
            result = self.executor.execute(conversation)
            feedback = self.check_response_integrity(result)
            if feedback == "":
                conversation.add_message(
                    {"role": "assistant", "content": result, "keep": True}
                )
                conversation.clean()
                break
            else:
                # TODO determine if we should keep the failed response or not
                logging.warning(f"Link execution re-attempted with feedback:{feedback}")
                conversation.add_message(
                    {"role": "user", "content": feedback, "keep": False}
                )
        return conversation

    def check_response_integrity(self, response) -> str:
        """Checks that that a generated response abides by the constraints of the link. If not, returns a feedback string for allowing the link's result to be improved.

        Args:
            response (str): Result from the link execution

        Raises:
            NotImplementedError: Abstract method

        Returns:
            str: Feedback string for allowing the link's result to be improved. If empty, the link's result is considered valid.
        """
        feedback = ""
        for constraint in self.constraints:
            feedback += constraint.check_response_integrity(response)
        return feedback

    def build_prompt(self, conversation: conversation.Conversation) -> str:
        """Builds the prompt for the link's execution

        Args:
            conversation (Conversation): Past conversation

        Raises:
            NotImplementedError: Abstract method

        Returns:
            str: Prompt for the link's execution
        """

        return jinja2.Template(self.prompt_template).render(conversation=conversation)
