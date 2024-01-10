"""
Implement the interface of applying GPT-based model to fragranceX data
"""

import os
import argparse

from chatbot.configure import api_keys, internal_prompt
from chatbot.data import read_frag_data_from_file

import openai
openai.api_key=api_keys[0]

from langchain.prompts.chat import AIMessage, HumanMessage, SystemMessage, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI


class GPT4Frag(object):
    def __init__(
        self, model_name: str, api_key: str, verbose: bool=True
    ) -> None:
        """
        :param model_name: The name of LLM,
        """

        # Which LLM to use
        self._model_name = model_name  # type: str

        # The large language model instance
        self._llm = None  # type: ChatOpenAI
        self._apikey = api_key  # type: str

        # "You are an expert"
        self._internal_prompt = SystemMessage(content=internal_prompt)

        # The latest LLM response
        self._llm_response = ""  # type: str

        # Log of the whole conversation
        self._global_conversations = []

        # The working directory of the agent
        self._path = "."  # type: str
        self._std_log = "description.log"  # type: str

        # The most recent error message
        self._errmsg = ""  # type: str
        self.verbose = verbose
        return
    

    @staticmethod
    def _log(info):
        print(info)
        # pass


    def _user_says(self) -> None:
        """
        Print header for user input in the log
        :return:
        """
        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("User says: ")
        self._global_conversations.append("------------------------\n")
        return


    def _chatbot_says(self) -> None:
        """
        Print header for LLM input in the log
        :return:
        """
        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("%s says: " % self._model_name)
        self._global_conversations.append("------------------------\n")
        return
    

    def _connect_chatbot(self) -> None:
        """
        Connect to chat bot
        :return:
        """

        self._llm = ChatOpenAI(
            model_name=self._model_name, temperature=0.3, openai_api_key=self._apikey
        )
        return
    

    def _get_frag_data(self) -> None:
        """
        Get problem data from description
        :return: The dictionary with problem and in/out information
        """

        self._data = read_frag_data_from_file(
            os.path.join(self._problem_path, self._std_format)
        )
    

    def dump_conversation(self, path_to_conversation: str = None) -> None:
        if path_to_conversation is None:
            path_to_conversation = self._std_log

        print("Dumping conversation to %s" % path_to_conversation)

        with open(path_to_conversation, "w") as f:
            f.write("\n".join(self._global_conversations))


    def talk_to_chatbot(self, user_input: str) -> str:

        conversation = [self._internal_prompt, formulation_request]

        self._log("==== Connecting to chat bot...")
        self._connect_chatbot()

        self._user_says()
        self._global_conversations.append(user_input)

        self._chatbot_says()
        output = self._llm(user_input)
        self._llm_response = output.content
        self._global_conversations.append(self._llm_response)

        if self.verbose:
            print(output.content)
            self.dump_conversation(os.path.join(self._path, self._std_log))
    
        return output.status




def read_args():
    """
    Read arguments from command line
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name")
    parser.add_argument(
        "--doc_path",
        type=str,
        default="fragrancex/fragrances/",
        help="Path to documents",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose mode")
    return parser.parse_args()


if __name__ == "__main__":
    # Read arguments from command line
    args = read_args()

    messages = [
        {'role': 'user', 'content': 'I want to buy a perfume for my wife.'},
    ]

    # GPT agent chatbot
    # agent = GPT4Frag(
    #     model_name=args.model,
    #     api_key=api_keys[0],
    #     verbose=args.verbose
    # )

    try:
        # status = agent.talk_to_chatbot()
        response = openai.ChatCompletion.create(
            model="gpt-3.5",
            messages=messages,
            temperature=0.6,
        )
        # print("Status: ", status)

    except Exception as e:
        raise e
    
    finally:
        pass
        # agent.dump_conversation(os.path.join(agent._path, agent._std_log))
