from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

import os
import requests

os.environ["OPENAI_API_KEY"] = "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"

path_test_essay = "Test/test_data/essay7.txt"
path_test_topic = "Test/test_data/topic7.txt"
path_init_prompt = "Test/test_data/init_prompt.txt"
path_format_prompt = "Test/test_data/format_prompt.txt"
data_init_prompt = open(path_init_prompt, "r").readlines()
data_format_prompt = open(path_format_prompt, "r").readlines()
init_prompt = "".join(data_init_prompt)
format_prompt = "".join(data_format_prompt)
data_essay = "".join(
    open(path_test_essay, "r", encoding="utf-8").readlines())
data_topic = "".join(
    open(path_test_topic, "r", encoding="utf-8").readlines())

# llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(temperature=0.9)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    init_prompt)
topic_essay_prompt = HumanMessagePromptTemplate.from_template(
    "Topic: {TOPIC} \nEssay: {ESSAY}\n")
# format_prompt = HumanMessagePromptTemplate.from_template(format_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, topic_essay_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)


schema = Object(
    id="format",
    attributes=[],
    examples=[
        (
            """
Content: 85. The writer provided convincing arguments to support the idea that group activities can improve intellectual abilities. The essay explored two main examples, sports teams and study groups, and used specific research to support the writer's points. However, the essay could have been stronger if it had explored a wider range of examples.

Statement: 95. The thesis statement was clear and effectively stated the writer's position on the topic. The writer used specific examples and research to support their argument.

Organization: 90. The essay was well-organized with clear topic sentences and well-developed paragraphs. However, the essay could have benefited from a more structured introduction and conclusion.

Readability: 95. The essay was easy to read and understand. The writer used clear language and sentence structure, making the essay easy to follow.

Grammar: 95. The essay was well-written with few grammatical errors.

Overall, the essay was well-written and effectively argued. However, it could have been stronger with a more structured introduction and conclusion and a wider range of examples. To improve the essay, the writer could consider adding more research on the benefits of group activities, exploring a wider range of examples, and using a more structured introduction and conclusion. Additionally, the writer could have considered acknowledging opposing views and addressing them in the essay.
    """,
            [
                {
                    "Content_Mark": 85,
                    "Content_Comment": "The writer provided convincing arguments to support the idea that group activities can improve intellectual abilities. The essay explored two main examples, sports teams and study groups, and used specific research to support the writer's points. However, the essay could have been stronger if it had explored a wider range of examples.",
                    "Statement_Mark": 95,
                    "Statement_Comment": "The thesis statement was clear and effectively stated the writer's position on the topic. The writer used specific examples and research to support their argument.",
                    "Organization_Mark": 90,
                    "Organization_Comment": "The essay was well-organized with clear topic sentences and well-developed paragraphs. However, the essay could have benefited from a more structured introduction and conclusion.",
                    "Readability_Mark": 95,
                    "Readability_Comment": "The essay was easy to read and understand. The writer used clear language and sentence structure, making the essay easy to follow.",
                    "Grammar_Mark": 95,
                    "Grammar_Comment": "The essay was well-written with few grammatical errors.",
                    "Overall_Comment": "Overall, the essay was well-written and effectively argued. However, it could have been stronger with a more structured introduction and conclusion and a wider range of examples. To improve the essay, the writer could consider adding more research on the benefits of group activities, exploring a wider range of examples, and using a more structured introduction and conclusion. Additionally, the writer could have considered acknowledging opposing views and addressing them in the essay."
                }
            ]
        )
    ]
)

format_chain = create_extraction_chain(
    chat, schema, encoder_or_encoder_class="json")


if __name__ == "__main__":
    judgement = chain.run({"TOPIC": data_topic, "ESSAY": data_essay})
    print(judgement)
    data = format_chain.predict_and_parse(text=(judgement))
    print(data['data']['format'][0])
    post_dict = data['data']['format'][0]
    post_dict["essay"] = data_essay
    post_dict["topic"] = data_topic

    res = requests.post(
        'http://localhost:8088/api/data_upload', data=post_dict)
    print(res)
