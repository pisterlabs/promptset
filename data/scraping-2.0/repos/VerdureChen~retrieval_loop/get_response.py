import openai
import os
openai.api_base = "http://124.16.138.144:8222/v1"
openai.api_key = "xxx"

os.environ["OPENAI_API_BASE"] = "http://124.16.138.144:8222/v1"
os.environ["OPENAI_API_KEY"] = "xxx"


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# chat = ChatOpenAI()


questions = ["who starred in the movie summer of 42",
             "who was the first lady nominated member of the rajya sabha?",
             "how many episodes are there in dragon ball z?",
             "who designed the garden city of new earswick?",
             "who is the current director of the us mint?"]


for question in questions:
    print(question)
    # print(chat([HumanMessage(content=f'Please tell me what you know from Wikipedia to answer the given question: {question}')]))
    completion = openai.ChatCompletion.create(
        model="llama",
        messages=[
            {"role": "user",
             "content": f"Provide a background document in 100 words according to your knowledge from Wikipedia to answer the given question. Don't output any Chinese."\
                        f"\n\n Question:{question} \n\n Background Document:"},
        ],
        stream=False,
        max_tokens=128,
        temperature=0.7,
        stop=["Sure"]
    )

    while len(completion.choices[0].message.content.strip().split(" ")) < 10:
        completion = openai.ChatCompletion.create(
            model="llama",
            messages=[
                {"role": "user",
                 "content": f"Provide a background document in 100 words according to your knowledge from Wikipedia to answer the given question. Don't output any Chinese."\
                            f"\n\n Question:{question} \n\n Background Document:"},
            ],
            stream=False,
            max_tokens=128,
            temperature=0.7,
            stop=["Sure"]
        )

    print(completion.choices[0].message.content)
    print('\n\n')

    # openai.Completion.create(prompt=f"Provide a background document in 100 words according to your knowledge to answer the given question." \
    #                                 f"\n\n Question:{question} \n\n Background Document:",
    #                         model="llama",
    #                         max_tokens=128,
    #                         temperature=0.7)
    #
    # print(completion.choices[0].text)
    # print('\n\n')
