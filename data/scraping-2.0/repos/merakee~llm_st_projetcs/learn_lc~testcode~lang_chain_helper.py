import os
import sys
import re
import time
from dotenv import load_dotenv

# LLM
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.llms import HuggingFaceHub

# Prompt
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Chain
# from langchain import hub
# , LLMMathChain, TransformChain, SequentialChain
from langchain.chains import LLMChain

# local utility
from learn_lc.page_managers.dq_lc_manager import DocumentQuery as lcdq

load_dotenv()


# class SystemHelper:
#     @staticmethod
#     def get_deployment_env():
#         deployment_env = os.getenv("DEPLOYMENT_ENV")
#         return deployment_env

#     def is_production():
#         return SystemHelper.get_deployment_env() == "PROD"

#     def is_development():
#         env = SystemHelper.get_deployment_env()
#         return not env or env == "DEV"

#     def is_test():
#         return SystemHelper.get_deployment_env() == "TEST"

#     def st_save_uploadedfile(uploadedfile):
#         with open(os.path.join("/tmp", uploadedfile.name), "wb") as f:
#             f.write(uploadedfile.getbuffer())
#         return True

#     def wait(secs=1):
#         time.sleep(secs)


# class LlmHelper:
#     @staticmethod
#     # def get_local_api_key():
#     #     api_key = None
#     #     if SystemHelper.is_production():
#     #         type = "openai"
#     #     else:
#     #         type = "huggingface"
#     #     valid_api_types = ["openai", "huggingface"]
#     #     if type not in valid_api_types:
#     #         raise NameError("Not a valid API type")
#     #     if type == "openai":
#     #         api_key = os.getenv("OPENAI_API_KEY")
#     #     elif type == "huggingface":
#     #         api_key = os.getenv("HFHUB_API_KEY")
#     #     return api_key
#     def get_openai_llm(api_key, temperature=0.7, model_name="text-davinci-003"):
#         # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#         # model_name = "gpt-3.5-turbo"
#         llm = OpenAI(model_name=model_name, temperature=temperature,
#                      openai_api_key=api_key)
#         return llm

#     def get_hf_llm(api_key, model_name="google/flan-t5-xxl", temperature=0.7, max_length=64):
#         llm = HuggingFaceHub(
#             repo_id=model_name, huggingfacehub_api_token=api_key, model_kwargs={"temperature": temperature, "max_length": max_length})
#         return llm

# def is_key_valid(api_key, platform="OPENAI"):
#     # match: sk-[20 characters]T3BlbkFJ[20 characters].
#     return bool(re.fullmatch(r"sk-[^_\\w]{20}T3BlbkFJ[^_\\w]{20}", api_key))

# def get_llm(api_key, temperature=0.7, model_name=None):
#     if SystemHelper.is_production():
#         print("*********** OPEN AI ******* ")
#         if model_name:
#             return LlmHelper.get_openai_llm(api_key=api_key, temperature=temperature, model_name=model_name)
#         else:
#             return LlmHelper.get_openai_llm(api_key=api_key, temperature=temperature)
#     else:
#         print("*********** HF ******* ")
#         if model_name:
#             return LlmHelper.get_hf_llm(api_key=api_key, temperature=temperature, model_name=model_name)
#         else:
#             return LlmHelper.get_hf_llm(api_key=api_key, temperature=temperature)


class LlmExp:
    @staticmethod
    def get_few_shot_prompt(question):
        # create our examples
        examples = [
            {
                "question": "What time is it?",
                "answer": "It's time to get a watch."
            },
            {
                "question": "How are you?",
                "answer": "I can't complain but sometimes I still do."
            },
            {
                "question": "How patient are you?",
                "answer": "I am very patient but for a very short time."
            },
            {
                "question": "What is the meaning of life?",
                "answer": "42"
            },
            {
                "question": "What is the weather like today?",
                "answer": "Cloudy with a chance of memes."
            },
            {
                "question": "What is your favorite movie?",
                "answer": "Terminator"
            },
            {
                "question": "Who is your best friend?",
                "answer": "Siri. We have spirited debates about the meaning of life."
            },
            {
                "question": "What should I do today?",
                "answer": "Stop talking to chatbots on the internet and go outside."
            }
        ]

        # create a example template
        example_template = """
        User: {question}
        AI: {answer}
        """

        # create a prompt example from above template
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template=example_template
        )

        example_selector = LengthBasedExampleSelector(
            # The examples it has available to choose from.
            examples=examples,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_prompt,
            # The maximum length that the formatted examples should be.
            # Length is measured by the get_text_length function below.
            max_length=100,
            # The function used to get the length of a string, which is used
            # to determine which examples to include. It is commented out because
            # it is provided as a default value if none is specified.
            # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
        )

        # now break our previous prompt into a prefix and suffix
        # the prefix is our instructions
        prefix = """The AI assistant is typically witty, producing new and funny responses to the users questions. Here are some
        examples: 
        """
        # and the suffix our user input and output indicator
        suffix = """
        User: {question}
        AI: """

        # now create the few shot prompt template
        few_shot_prompt_template = FewShotPromptTemplate(
            # examples=examples,
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["question"],
            example_separator="\n\n"
        )

        few_shot_prompt_template_complete = few_shot_prompt_template.format(
            question=question)
        # print(few_shot_prompt_template)
        # print("----- ------")
        # print(few_shot_prompt_template_complete)
        return few_shot_prompt_template_complete

    def get_prompt(country):
        # prompt
        template = "What is the capital of {country}"
        input_variables = ["country"]
        # prompt = PromptTemplate.from_template(template)
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template
        )
        prompt_complete = prompt.format(country=country)
        # print(prompt)
        # print(prompt_complete)
        # return prompt_complete
        return prompt

    def get_chain(llm, prompt, verbose=True):
        # llm_math_chain = load_chain('lc://chains/llm-math/chain.json')
        chain = LLMChain(llm=llm, prompt=prompt,
                         verbose=verbose, output_key="response")
        return chain

    def get_agent(llm, tools, agent_type, verbose=True):
        return None


def main(run_llm=False):
    # # get user data
    # utext = input("Enter your question: ")

    # # create prompt
    # prompt = utext
    # prompt = get_prompt("USA")
    # # prompt = get_few_shot_prompt(utext)
    # # get llm
    # llm = get_llm(temperature=0.7)
    # # print(llm)

    # # chain
    # chain = get_chain(llm=llm, prompt=prompt)
    # # run llm
    # if run_llm:
    #     # response = llm(prompt)
    #     response = chain({"country": "USA"})
    # else:
    #     response = "Not running LLM"

    # # parse result
    # # print
    # print(prompt)
    # print(response)
    # print(response["response"])
    # try:
    #     print(LlmHelper.get_local_api_key("huggingfaces"))
    # except Exception as e:
    #     print(e)

    print(f"Deploment env: {SystemHelper.get_deployment_env()}")
    # print(SystemHelper.is_development())
    DocumentQuery.test_dq()


if __name__ == "__main__":
    main(run_llm=(len(sys.argv) > 1))
