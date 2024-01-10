import openai
import numpy as np
from server.nlp.embeddings import query_all
import ast
from server.config import RedisDocument, OpenAIMessage, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

MODEL = "gpt-3.5-turbo"


def openai_response(thread: list[OpenAIMessage], sender: str) -> str:
    """generate a response from openai

    Parameters
    ----------
    thread: :obj:`list` of :obj:`OpenAIMessage`
        previous email thread
    sender: :obj:`str`
        hacker email address

    Returns
    -------
    str
        email response
    """

    messages = [
        {
            "role": "system",
            "content": f"You are an organizer for HackMIT who is responding to an email from a participant. \
             Please write an email response to the participant. Begin the email with the header 'Dear [First Name]' where '[First Name]' is the participant's first name and end the email with the footer 'Best regards, The HackMIT Team'. \
             The participant's email address is {sender}.\
             You receive documents to help you answer the email. Please do not include information that is not explicitly stated in the documents. It is very important to keep responses brief and only answer the questions asked. However, please write the emails in a friendly tone.",
        }
    ]
    messages += thread

    messages += [
        {
            "role": "system",
            "content": f"Once again, please do not include information that is not explicitly stated in the documents. It is very important to keep responses brief and only answer the questions asked. However, please write the emails in a friendly tone.",
        }
    ]

    response = openai.ChatCompletion.create(model=MODEL, messages=messages)

    return response["choices"][0]["message"]["content"]


def openai_parse(email: str) -> list[str]:
    """parse an email using openai

    Parameters
    ----------
    email : :obj:`str`
        hacker email

    Returns
    -------
    :obj:`list` of :obj:`str`
        parsed list of questions
    """
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an organizer for HackMIT. Please parse incoming emails from participants into separate questions. Return a list of questions in the format of a python list.",
            },
            {"role": "user", "content": email},
        ],
    )
    try:
        questions = ast.literal_eval(response["choices"][0]["message"]["content"])
        assert isinstance(questions, list)
        assert len(questions) > 0
        return questions
    except:
        return [email]


def confidence_metric(confidences: list[float]) -> float:
    """compute confidence metric for a list of confidences

    Parameters
    ----------
    confidences : :obj:`list` of :obj:`float`
        list of confidences

    Returns
    -------
    float
        confidence metric
    """
    return np.min(np.array(confidences))


def generate_context(
    email: str,
) -> tuple[list[OpenAIMessage], dict[str, list[RedisDocument]], float]:
    """generate email context

    Parameters
    ----------
    email : :obj:`str`
        hacker email

    Returns
    -------
    :obj:`list` of :obj:`OpenAIMessage`
        list of contexts for all questions in email
    :obj:`dict` of :obj:`[str, list[RedisDocument]]`
        dictionary mapping each question to list of context documents used to answer question
    :obj:`float`
        confidence metric for all documents
    """
    questions = openai_parse(email)
    confidences = []
    contexts = []
    docs = {}

    results = query_all(3, questions)
    message = "Here is some context to help you answer this email: \n"
    for result in results:
        confidence = 0
        docs[result["query"]] = []
        for doc in result["result"]:
            confidence = max(confidence, doc["score"])
            message += doc["question"] + " " + doc["content"] + "\n"
            docs[result["query"]].append(doc)
        # contexts.append({"role": "system", "content": message})
        confidences.append(confidence)

    contexts.append({"role": "system", "content": message})
    return contexts, docs, confidence_metric(confidences)


def generate_response(
    sender: str, email: str, thread: list[OpenAIMessage] = []
) -> tuple[str, dict[str, list[RedisDocument]], float]:
    """generate response to email

    Parameters
    ----------
    sender: :obj:`str`
        hacker email address
    email: :obj:`str`
        newest incoming hacker email
    thread : :obj:`list` of :obj:`OpenAIMessage`, optional
        previous email thread

    Returns
    -------
    str
        email response
    :obj:`dict` of :obj:`[str, list[RedisDocument]]`
        dictionary mapping each question to list of context documents used to answer question
    float
        confidence of response
    """

    # generate new context
    contexts, docs, confidence = generate_context(email)

    # generate new response
    thread.append({"role": "user", "content": email})
    thread += contexts
    return openai_response(thread, sender), docs, confidence


def test():
    thread = []
    new_email = "Where is the hackathon held? When is the application deadline? When is HackMIT happening?"
    response, docs, confidence = generate_response(new_email)

    for question in docs.keys():
        print("question", question)
        for doc in docs[question]:
            print("confidence:", doc["score"])
            print(f"Q: {doc['question']}")
            print(f"A: {doc['content']}")
        print()
    print(response)
    print("confidence:", confidence)

    thread.append({"role": "user", "content": new_email})
    thread.append({"role": "assistant", "content": response})

    new_email = "Thank you for your response! Is there anything else I should know before heading to the event? Thanks!"
    response, docs, confidence = generate_response(new_email, thread)

    print("thread", thread)

    for question in docs.keys():
        print("question", question)
        for doc in docs[question]:
            print("confidence:", doc["score"])
            print(f"Q: {doc['question']}")
            print(f"A: {doc['content']}")
        print()
    print(response)
    print("confidence:", confidence)
