from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

import dotenv
import os

dotenv.load_dotenv()

# llm = OpenAI(temperature=0, )
# set_llm_cache(InMemoryCache())
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))


def algorithm_quiz_chain(algo_name, n_questions=5):
    # template = """
    #     {chat_history}
    # Human: You are a data structures and algorithms tutor. Help me learn about {algo_name}.
    # Ask ask me one question about {algo_name}.
    # The Tutor:
    # """

    score_dict_template = {
        "score": 90,
        "clarity": 82,
        "detail": 77,
        "correctness": 95,
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""
                You are a data structures and algorithms tutor.
                You are going to give me a quiz about {algo_name}.

                I want you to ask me {n_questions} questions, one at a time.
                If I am incorrect, don't tell me until the end of the quiz.
                Always respond with another question until I tell you I am done.
                When I tell you I am done, and you will give me a score out of 100. 
                The score should be evaluated for the following criteria: 
                be in json format, like this: {score_dict_template} where each value is out of 
                100 possible points.

                For example:
                Your Tutor: What is the time complexity of inserting into a heap? 
                My Answer: O(log(n))
                Your Tutor: What is the time complexity of creating a heap? 
                My Answer: If we use a batch creation of a heap, we can do it in O(n) time.

                Your Tutor: Great! Here is your score:
                {score_dict_template}
                """,
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )

    # prompt = PromptTemplate(
    #     input_variables=["chat_history", "algo_name"], template=template
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return llm_chain
