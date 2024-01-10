import datetime
import json
import openai
import re
import time

from question import Question

from typing import Dict, Iterable, List, Optional, Union

sample_input = """
Dear Santa Claus, My name is Yadiel and I am 4 years old. I'm from Dominican parents, but I borned in the United States. I wish you to give me something for Chritsmas. My parents do not have enough money for buy me something. My dad is the only one that is working and my mom is pregnant. My sister, Yazlyn, will born is Chritsmas and I will love if you send her something too for Chritsmas. It will mean something big to me if you send her something. My sizes in clothes are the following: coats, t-shirts, swetters: 4t. Pants, pajamas, and interior clothes: 4t. Sneakers, boots and shoes: 11.5. I am a little friendfull (friendly) and loving boy. I've been a good boy this whole year. I got good news for you. I can sleep without doing pee in my bed since June. With Love, Yadiel.
"""


def send_gpt_chat(
    messages: Union[str, Iterable],
    *,
    openai_client: openai.OpenAI,
    model: str,
    timeout: Union[float, openai.Timeout, None] = None,
    retries: int = 3,
    throttle: float = 3.0,
):
    if type(messages) == str:
        messages = [{"role": "user", "content": messages}]

    while retries > 0:
        retries -= 1
        try:
            response = openai_client.chat.completions.create(
                messages=messages, model=model, temperature=0, timeout=timeout
            )
            if not response or not response.choices or not len(response.choices):
                return None
            if response.choices[0].finish_reason != "stop":
                return None
            return response.choices[0].message.content

        except openai.APITimeoutError:
            pass
        except openai.InternalServerError:
            pass
        except openai.RateLimitError:
            pass

        if throttle:
            time.sleep(throttle)


def create_systemprompt(question: Question, document_description: str = None) -> str:
    systemprompt = ""

    systemprompt += (
        "I will present a short document to you. You will read this document "
        "and then extract a single piece of information from that document. "
        "You will be graded on your reasoning process and your ability to "
        "justify your answer.\n\n"
    )

    if document_description:
        systemprompt += f"The document can be described as: {document_description}\n\n"

    systemprompt += f"""
The piece of information I'd like you to extract is: {question.text}

Present your response in Markdown format, using the following multi-part structure: RELEVANCE, AVAILABILITY, DISCUSSION, and ANSWER. Each part will begin with its header, followed by your content.

# RELEVANCE
Here, you will determine whether or not the desired piece of information is relevant to the subject matter of the document. You will ultimately write, in all caps, either RELEVANT (it's relevant), or OFFTOPIC (it's off-topic).

# AVAILABILITY
Here, you will determine whether or not the desired information is present in the document. You will ultimately write, in all caps, one of the following: STATED (the information is explicitly stated in the document), IMPLIED (the information is implied by other content in the document), or ABSENT (the information cannot be determined from the document).

# COMPUTATION
If the problem requires any kind of counting, enumeration, calculation, or so forth, then you can use this section as a scratchpad upon which to work out your math. If the problem doesn't require any such processes, then you can simply skip this section if you wish.

# DISCUSSION
Here, you will discuss what your final answer will be. You will give arguments about why the answer might be one thing or another.

# ANSWER
Here, you will state your final answer in a succinct manner, with no other text, as if you are going to enter the value into a form.

"""

    if question.datatype is not None:
        systemprompt += "You will present your final answer in the following format: "
        systemprompt += question.instructions_for_my_datatype()
        systemprompt += "\n\n"

    if question.required:
        systemprompt += "It is mandatory that you provide *some* answer in the ANSWER section. If needed, just take your best guess.\n\n"

    systemprompt += "Good luck."

    return systemprompt


def split_gpt_output(gpt_output):
    matches = re.findall(r"# (.*?)\n(.*?)(?=# |\Z)", gpt_output, re.DOTALL)

    retval = {match[0]: match[1].strip() for match in matches}
    return retval


def extract_gpt_answer(gpt_output):
    outdict = split_gpt_output(gpt_output)

    has_relevant_token = "RELEVANT" in outdict.get("RELEVANCE", "")
    has_offtopic_token = "OFFTOPIC" in outdict.get("RELEVANCE", "")
    if (not has_relevant_token and not has_offtopic_token) or (
        has_relevant_token and has_offtopic_token
    ):
        raise ValueError("Can't have both (or neither) for RELEVANCE")

    if has_offtopic_token:
        return None

    has_absent_token = "ABSENT" in outdict.get("AVAILABILITY", "")
    if has_absent_token:
        return None

    answer = outdict.get("ANSWER")
    return answer


def ask_gpt_question(question, document, document_description):
    sysprompt = create_systemprompt(
        question=question, document_description=document_description
    )

    gpt_messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": document},
    ]

    responseobj = openai.ChatCompletion.create(
        messages=gpt_messages, model="gpt-4", temperature=0
    )
    # TODO: Check for errors and wrap this in retries.
    gpt_output = responseobj["choices"][0]["message"]["content"]
    answer = extract_gpt_answer(gpt_output)

    return answer


def extract_dict_from_document(
    document: str, questions: Iterable[str], document_description: str = None
):
    retval = {}
    for k, v in questions.items():
        print(k, end="")
        answer = ask_gpt_question(v, document, document_description)
        retval[k] = answer
        print(answer)
    return retval


def determine_datatypes(
    questions: List[Question],
    *,
    openai_client: openai.OpenAI,
    document_description: Optional[str] = None,
) -> List[Question]:
    if type(document_description) == tuple:
        document_description = document_description[0]

    prompt = (
        "I'm a programmer who's writing a data ingestion script for a client. "
        "I need your help to determine the best variable types with which to represent "
        "the data that the client wants to extract.\n\n"
        "The client will give me a series of documents. I haven't seen the documents myself "
        "yet, but I know that there will be hundreds of them. "
    )
    if document_description:
        prompt += "Each document can best be described as: " + document_description

    prompt += "\n\nFrom each document, I need to extract the following variables:\n\n"

    for question in questions:
        prompt += f"- **{question.key}**: {question.text}\n"

    prompt += (
        "\nI need to pick an appropriate data type for each variable. "
        "I have a fixed set of data types at my disposal. The data types I can use "
        "are as follows:\n\n"
        "- **int**\n"
        "- **float**\n"
        "- **str**\n"
        "- **List[int]** (i.e. a list of integers)\n"
        "- **List[float]** (i.e. a list of floats)\n"
        "- **List[str]** (i.e. a list of strings)\n"
        "- **date** (i.e. a Python datetime.date object)\n"
        "- **datetime** (i.e. a Python datetime.datetime object)\n"
        "- **timedelta** (i.e. a Python datetime.timedelta object)\n"
        '- **enum("VALUE_1", "VALUE_2", ...)** (i.e. an enum with a set number of possible values, each of which is denoted with a string)\n'
        "\nFor numerical data types, I also have the option to provide a string that indicates the number's units.\n\n"
    )

    prompt += (
        "I'd like you to go through each variable, one at a time, and determine which of the "
        "above data types would be most appropriate for it. You will provide the name of the "
        "variable, a brief discussion about what its best data type might be, a datatype, and "
        "a unit label (if appropriate). In some cases, you might even choose an appropriate default value. "
        "As such, for each variable, your reply will look like this:\n"
        "\n"
        "VARIABLE: name_of_variable\n"
        "DISCUSSION: Here you discuss which of the available data types would best suit this variable.\n"
        "DATATYPE: one of the above data types\n"
        "UNITS: for numerical types, a label indicating what units the variable's value will represent\n"
        "DEFAULT: a default value, if one seems appropriate\n"
        "\n"
        "Here are a few examples:\n"
        "\n"
        "VARIABLE: bank_account_balance\n"
        "DISCUSSION: A bank account is represented by a scalar numerical value. We don't know the currency, "
        "so we will assume USD because it's the most commonly used currency in the world. "
        "To represent cents, we need decimal support; as such, a floating-point value is the most "
        "appropriate choice. As for default value, we'll choose a round number for a typical "
        "checking account balance.\n"
        "DATATYPE: float\n"
        "UNITS: U.S. Dollars (US$)\n"
        "DEFAULT: 10000.0\n"
        "\n"
        "VARIABLE: us_coin\n"
        "DISCUSSION: The US Mint only makes a few denominations of coins, so an enum would be the most appropriate.\n"
        'DATATYPE: enum("PENNY", "NICKEL", "DIME", "QUARTER", "HALFDOLLAR", "SILVERDOLLAR")\n'
        "UNITS: N/A\n"
        "DEFAULT: N/A"
    )

    # The timeout should be proportional to the number of questions.
    # Each question really shouldn't take more than five seconds max
    # to determine the data type.
    timeout = 10 + 5 * len(questions)

    reply = send_gpt_chat(
        messages=prompt,
        timeout=timeout,
        model="gpt-3.5-turbo-16k",
        openai_client=openai_client,
    )

    reply_lines = reply.split("\n")
    q_by_key = {q.key: q for q in questions}
    q_current = None
    for line in reply_lines:
        if ":" not in line:
            continue
        line = line.strip()
        fieldname, fieldvalue = line.split(":", maxsplit=1)
        fieldname = fieldname.strip()
        fieldvalue = fieldvalue.strip()

        if fieldvalue.upper() == "N/A":
            continue

        if fieldname.upper() == "VARIABLE":
            q_current = q_by_key.get(fieldvalue)
            continue

        if not q_current:
            continue

        if fieldname.upper() == "DISCUSSION":
            if not q_current.explanation:
                q_current.explanation = fieldvalue

        elif fieldname.upper() == "UNITS":
            if not q_current.unitlabel:
                q_current.unitlabel = fieldvalue

        elif fieldname.upper() == "DEFAULT":
            if not q_current.defaultvalue:
                q_current.defaultvalue = fieldvalue

        elif fieldname.upper() == "DATATYPE":
            if not q_current.datatype:
                if fieldvalue == "int":
                    q_current.datatype = int
                elif fieldvalue == "float":
                    q_current.datatype = float
                elif fieldvalue == "str":
                    q_current.datatype = str
                if fieldvalue == "List[int]":
                    q_current.datatype = List[str]
                elif fieldvalue == "List[float]":
                    q_current.datatype = List[float]
                elif fieldvalue == "List[str]":
                    q_current.datatype = List[str]
                elif fieldvalue == "date":
                    q_current.datatype = datetime.date
                elif fieldvalue == "datetime":
                    q_current.datatype = datetime.datetime
                elif fieldvalue == "timedelta":
                    q_current.datatype = datetime.timedelta
                elif fieldvalue.startswith("enum"):
                    valueliststr = "[" + fieldvalue[5:-1] + "]"
                    try:
                        q_current.datatype = json.loads(valueliststr)
                    except:
                        pass

    for q in questions:
        if q.defaultvalue is not None and q.datatype is not None:
            q.defaultvalue = q.coerce_to_my_datatype(q.defaultvalue)

    return questions


SECRETS = {}
with open("secrets.json") as f:
    SECRETS = json.load(f)

openai_client = openai.OpenAI(
    api_key=SECRETS["OPENAI_API_KEY"], organization=SECRETS.get("OPENAI_ORGANIZATION")
)


document_description = "A letter from a child to Santa Claus"


questions = dict(
    name="What is the child's name?",
    age="How old are they?",
    wealth={
        "text": "What socioeconomic bracket are they in?",
        "datatype": ["POOR", "MIDDLECLASS", "RICH"],
    },
    present_desired="What present or presents do they want?",
    misspellings_count="How many misspellings or grammatical mistakes did they make?",
)
questions = Question.create_collection(questions)

questions = determine_datatypes(
    questions=questions,
    document_description=document_description,
    openai_client=openai_client,
)

for q in questions:
    print(q)

# retval = extract_dict_from_document(
#    sample_input,
#    questions=questions,
#    document_description=document_description
# )
