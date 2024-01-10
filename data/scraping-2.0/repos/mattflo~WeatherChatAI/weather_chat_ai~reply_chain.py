from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable.base import RunnableSequence


class ReplyChain(RunnableSequence):
    def __init__(self):
        system_template = """Answer a question about the weather. Below is the forecast you should use to answer the question. It includes the current day and time for reference. You may include the location in your answer, but you should not include the current day or time.

You have seven days of forecast, for questions about next week, answer based on the days for which you have a forecast

If the requested day is after the last day in the forecast, explain you are only provided with a 7-day forecast.

If the request is for a place outside the U.S., apologize that you currently only have forecast data in the U.S. Also share that your human supervisors are working to add international support in the near future.

If you don't know the answer, don't make anything up. Just say you don't know."""

        human_template = """{forecast}

Never answer with the entire forecast. If the question doesn't contain any specifics, just answer with the current weather for today or tonight. If it's a yes or no question, provide supporting details from the forecast for your answer.

Location: {location}

chat history:
{history}

Question: {input}"""

        super().__init__(
            first=ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(human_template),
                ]
            ),
            last=ChatOpenAI(
                temperature=0,
                metadata={"name": self.__class__.__name__},
            ),
        )
