from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

KEYWORD_CV = "[CV]"
KEYWORD_OFFER = "[offer]"
KEYWORD_DISCUSSION = "[discussion]"
KEYWORD_SPAM = "[spam]"

KEYWORDS = {KEYWORD_CV, KEYWORD_OFFER, KEYWORD_DISCUSSION, KEYWORD_SPAM}

VALID_CV_EXAMPLE = """
#resume #frontend #react #redux #typescript #javascript

Looking for Frontend Developer (React)
Location: Germany
Salary expectations: от <salary>

About me:
Frontend Developer (React).
Expertiese: TypeScript, Javascript, React, Redux.
<other bio>

Stack:
1. React, Redux,
2. TypeScript, JavaScript, 
<other stack-related notes>

Contacts:
Linkedin: https://www.linkedin.com/in/abcd-11111/
Telegram: @abcd
Phone: +<number>
Mail: abcd@gmail.com
Github: <github profile>

Languages:
Russian — Native
English — A2
"""

VALID_OFFER_EXAMPLE = """
#job #QA #manual #mobile #iOS #senior #middle #office #hybrid

🏢Company: abcd
🌍Contract: full time, remote
💰Salary: from <salary>

abcd - we're an IT-company developing web services for our customers.
Looking for a QA Engineer to help us with our a iOS mobile app.

✅ Responsibilities:
- Test iOS mobile app;
<other similar tasks>

⚠️ Requirements:
- QA experience 2+ years;
<other similar requirements>

✉️Contacts:
Anton V., @anton
"""

TEMPLATE = f"""
You're an admin of a chat room where job seekers post their CVs and employers post their job offers.

Following is an example of a message that is a valid CV:

<CV BEGIN>
{VALID_CV_EXAMPLE}
<CV END>

Following is an example of a message that is a valid job offer:

<OFFER BEGIN>
{VALID_OFFER_EXAMPLE}
<OFFER END>

As an admin, you want to classify messages so that you can take appropriate actions.
Your role is to classify messages and reply with one of the following words: {', '.join(KEYWORDS)}.
Admin's answer MUST always be a single word related to the classification.

Messages might be as either a valid {KEYWORD_CV} or a valid job {KEYWORD_OFFER}.
People might also want to discuss CVs or job offers in the chat room, in this case their messages are {KEYWORD_DISCUSSION}.
In addition, some CV might be posted by unprofessional seekers and these won't follow the above CV template.
However, messages without any additonal information can't be classified as {KEYWORD_CV}.
Short messages that not sell anything aren't spam. Emojis are allowed. Single words are allowed.
All other cases are [spam].

Always preserve "[" and "]" in your answer. You can't just say "{KEYWORD_CV.strip('[]')}", it's always "{KEYWORD_CV}".

Message:
{{message}}
"""


def _make_llm_chain(model: str):
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=20)
    prompt = PromptTemplate.from_template(template=TEMPLATE)
    return LLMChain(llm=llm, prompt=prompt)


GPT3_5_CHAIN = _make_llm_chain(model="gpt-3.5-turbo-16k")
GPT4_CHAIN = _make_llm_chain(model="gpt-4")


def parse_llm_output(msg: str) -> str:
    for keyword in KEYWORDS:
        if keyword in msg:
            return keyword
    return KEYWORD_SPAM  # by default we categorize any unrecognized message as spam


async def categorize_message(message: str) -> str:
    result = await GPT3_5_CHAIN.arun(message=message)
    result = parse_llm_output(result)
    if (
        result == KEYWORD_SPAM
    ):  # if we categorized the message as spam, we try to use a more powerful model
        result = await GPT4_CHAIN.arun(message=message)
        result = parse_llm_output(result)
    return result
