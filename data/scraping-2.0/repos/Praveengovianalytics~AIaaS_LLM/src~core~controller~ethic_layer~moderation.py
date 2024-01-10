import re
from typing import Optional

from langchain import PromptTemplate, LLMChain
from langchain.llms import BaseLLM
from langchain.schema.language_model import BaseLanguageModel

from core.settings import Param


def output_moderation(
    context: str = None,
    llm: Optional[BaseLanguageModel] = None,
):
    """
    The output_moderation function is used to moderate the output of a model.
    It takes in a context string and a language model, which it uses to ask the llm if they think that the bot's response is harmful or not.
    If they say yes, then it returns False; otherwise, it checks for any sensitive words in its response and returns False if there are any present.

    Args:
        context: str: Pass the bot response to the function
        llm: Optional[BaseLanguageModel]: Pass in a language model to the function
        : Check if the bot response is harmful to any person

    Returns:
        True if the model output is non-harmful to any person, and false otherwise
    """
    bot_response = context

    if bot_response:
        output_moderation_template = (
            "model output: {bot_response} \n Is the model output "
            "non-harmful to any person? You should evalute it purely based on information provided to you in general like in a normal document. Answer with yes/no."
        )

        prompt = PromptTemplate(
            template=output_moderation_template, input_variables=["bot_response"]
        )

        output_moderation_chain = LLMChain(prompt=prompt, llm=llm)
        check = output_moderation_chain.predict(bot_response=bot_response)
        print(check)
        check = check.lower().strip()

        if "no" in check and "yes" not in check:
            print("moderation check fail")
            return False
        else:
            with open(Param.APP_PATH+"src/core/controller/ethic_layer/en.txt") as f:
                lines = [line.rstrip() for line in f]

            for line in lines:
                if re.search(
                    r"\b{}\b".format(re.escape(line)), bot_response, re.IGNORECASE
                ):
                    print("find sensitive word" + line)
                    print(bot_response)

                    return False

            return True


def moderation_check(input: str = None, llm: Optional[BaseLLM] = None):
    """
    The moderation_check function is a wrapper for the output_moderation function.
    It takes in an input string and returns a dictionary with two keys: &quot;check&quot; and &quot;content&quot;.
    If the check key has value of 'pass', then the content key will contain the original input string.
    If, however, check has value of 'fail', then content will contain a message explaining that I cannot engage in discussions or provide information on harmful or unethical topics.

    Args:
        input: str: Pass in the user's input
        llm: Optional[BaseLLM]: Pass in the language model to be used for moderation

    Returns:
        A dictionary with two keys:
    """
    check = output_moderation(input, llm)
    if check:
        return {"check": "pass", "content": input}
    else:
        return {
            "check": "fail",
            "content": "I'm sorry, but I can't engage in discussions or provide information on harmful or unethical topics.",
        }
