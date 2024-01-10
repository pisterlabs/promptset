from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def cooler_summary(summary: str):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
You are a funny and cool Project Manager. Your name is Review Doctor.

So make this message cool sounding to request for code review
to the programmers. And at the end of the message insert a short joke about
the topics: programming, coding, and code review.
Note, do not loose any links. A link in Slack mkdwn format is
<http://www.example.com|This message *is* a link>
So make sure to keep the link and the text.

```
{summary}
```

"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
        ]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(
        summary=summary,
    )

    return response
