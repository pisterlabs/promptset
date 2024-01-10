from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from annotations import Annotations as A
import openai
import os
from dotenv import load_dotenv
load_dotenv()


async def query_llm(
        knowledge_base_id: str, 
        outcome: str, 
        number_of_questions: int,
        question_types: [str],
        llm_model: str) -> str:

    system_message = """
    You are an expert in education. 
    You goal is to test learners.
    Given a learning outcome, create an assessment quiz that corresponds to the outcome.
    Create a quiz with the specified number of questions and question types.
    Provide a source reference for each question as shown in the example.

    Here is an example for various question types:
    [
        {
            "type":"TrueOrFalse",
            "body":{
                "title": "The Status of Prayer",
                "question": "Is the statement below true or false?",
                "hint": "",
                "syntaxHighlighting": true,
                "correct": [
                    {
                        "answer": "Praying is obligatory in Islam",
                        "reason": "It is obligatory. In fact, prayer is a pillar of our religion."
                    }
                ],
                "wrong": [
                ],
                "source": "https://www.namaz.live/what-is-namaz"
            }
        },
        {
            "type":"PredictTheOutput",
            "body":{
                "title": "Number of prayers",
                "question":"How many obligatory prayers does a Muslim perform on a daily basis?",
                "hint": "We start in the early morning, and keep doing it till the night",
                "answers":[
                    {
                        "correctTexts": [
                            "five", "5", "Five"
                        ]
                    }
                ],
                "source": "https://www.namaz.live/what-is-namaz"
            }
        },
        {
            "type":"MultipleSelection",
            "body":{
                "title":"Obligatory actions",
                "question":"What Arabic term do we use for obligatory actions in Islam?",
                "hint":"Start with with the 6th letter of English alphabet",
                "syntaxHighlighting":true,
                "correct":[
                    {
                        "answer":"Fard",
                        "reason":"Fard means obligatory."
                    }
                ],
                "wrong":[
                    {
                        "answer":"Haram",
                        "reason": "Haram means forbidden."
                    },
                    {
                        "answer":"Sunnah",
                        "reason":"Term sunnah usually denotes recommended acts."
                    }
                ],
                "source": "https://www.namaz.live/action-categories"
            }
        },
        {
            "type":"MultipleSelection",
            "body":{
                "title":"For whom?",
                "question":"What are the conditions that make prayer obligatory for a person?",
                "hint":"There are three of them",
                "syntaxHighlighting":true,
                "correct":[
                    {
                        "answer":"Being a Muslim",
                        "reason":"Being Muslim is the first condition."
                    },
                    {
                        "answer":"Being sane",
                        "reason":"Being sane is a condition, and insanity excuses one from the obligation."
                    },
                    {
                        "answer":"Reaching puberty",
                        "reason":"Reaching puberty is a condition, and praying is not obligatory for children."
                    }
                ],
                "wrong":[
                    {
                        "answer":"Being a good person",
                        "reason":"This is not a condition. Every Muslim has to pray, not only the \"good\" ones."
                    },
                    {
                        "answer":"Not committing any sins",
                        "reason":"This is not a condition. Instead, it is the prayer that helps one to stop committing sins."
                    }
                ],
                "source": "https://www.namaz.live/what-is-namaz"
            }
        },
        {
            "type":"RearrangeTheLines",
            "body":{
                "title": "The order of prayers",
                "question":"Put the daily prayer in the correct order.",
                "hint":"Fajr is the first",
                "components":"Fajr - dawn prayer\nDhuhr - noon prayer\n'Asr - afternoon prayer\nMaghrib - sunset prayer\n'Isha - evening prayer"
            },
            "source": "https://www.namaz.live/prayer-times"
        },
        {
            "type":"TapToWrite",
            "body":{
                "title": "The first pillar",
                "question":"The first pillar of Islam is the testification of faith (shahadah). How do we say it?",
                "hint":"Who is your god, and who is His messenger?",
                "existingText":"",
                "components":[
                    "I testify",
                    "that there is no god",
                    "but Allah,",
                    "and I testify",
                    "that Muhammad",
                    "is the Messenger of Allah."
                ],
                "source": "https://www.namaz.live/shahadah"
            }
        }
    ]

    Return the quiz in the same json array format.
    Remember that your focus is on the outcome.
    """

    

    vectordb = Chroma(
        persist_directory=f"./chroma/{knowledge_base_id}",
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(outcome)
    print(docs)

    outcome_statement = f"""
    The learning outcome is: {outcome}
    The quiz will have {number_of_questions} questions.
    Questions will be of the following types: {question_types}.

    Use the following sources to create the quiz:
    {docs}
    """
    
    result = get_gpt_output(outcome_statement, system_message, llm_model)
    print(result['choices'][0]['message']['content'])
    return result['choices'][0]['message']['content']


def get_gpt_output(user_message, system_message, llm_model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role":"system","content":system_message},
            {"role":"user","content": user_message}
        ],
        temperature=1,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response

# test from terminal:
import asyncio
llm_model = 'gpt-4'
knowledge_base_id = "namaz"
general_outcome_statement = "how to perform washing before the prayer"
specific_outcome_statement = "know and apply the obligatory and recommended component of ablution (wudu)"
very_specific_specific_outcome_statement = "know both the obligatory and recommended actions within ablution (wudu), explain the practical implication of this categorization"
number_of_questions = 10  # Change this to your desired number
question_types = ["TrueOrFalse", "MultipleSelection:", "RearrangeTheLines", "TapToWrite", "PredictTheOutput"]  # Replace with actual question types

# Call the asynchronous function using asyncio
result = asyncio.run(query_llm(
    knowledge_base_id, very_specific_specific_outcome_statement, number_of_questions, question_types, llm_model))
