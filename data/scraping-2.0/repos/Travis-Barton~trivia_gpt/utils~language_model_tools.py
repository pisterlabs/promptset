import os
from typing import List, Dict, Union
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.utilities import BingSearchAPIWrapper
from typing import List
import toml
from langchain.agents import initialize_agent, AgentType, Tool, LLMSingleActionAgent, AgentExecutor, AgentOutputParser
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper, SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
import asyncio
from langchain.schema import AgentAction, AgentFinish
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from concurrent.futures import ThreadPoolExecutor
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import os
import toml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the secrets.toml file
SECRETS_PATH = os.path.join(BASE_DIR, '..', '.streamlit', 'secrets.toml')

# Load the secrets from the file
toml_secrets = toml.load(SECRETS_PATH)
try:
    toml_secrets = toml.load(SECRETS_PATH)
    os.environ["BING_SUBSCRIPTION_KEY"] = toml_secrets['llm_api_keys']['BING_SEARCH_API']
    os.environ['OPENAI_API_KEY'] = toml_secrets['llm_api_keys']['OPENAI_API_KEY']
    os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
except FileNotFoundError:
    print(f"Could not find the secrets file at: {SECRETS_PATH}")
    # You can either exit the program or fall back to other methods for obtaining secrets
    # For now, we'll just print the error. You can decide what action to take.

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = toml_secrets['langsmith_api']['langsmith_api']
os.environ['LANGCHAIN_PROJECT'] = 'trivia-gpt'


class Question(BaseModel):
    question: str = Field(description="The trivia question")
    answer: str = Field(description="answer to the trivia question")
    category: str = Field(description="category of the question")
    difficulty: str = Field(description="difficulty of the question")  # options: easy, medium, hard


class QuestionSet(BaseModel):
    questions: List[Question] = Field(description="A list of questions for each category")


class FactCheckQuestion(BaseModel):
    question: str = Field(description="The trivia question")
    answer: str = Field(description="answer to the trivia question provided by the user")
    category: str = Field(description="category of the question provided by the user")
    fact_check: bool = Field(description="whether the answer is correct or not")
    explanation: str = Field(description="comment on the answer provided by the user")


def get_prompt(section, value):
    try:
        prompts = toml.load('prompts.toml')
    except:
        prompts = toml.load('../prompts.toml')
    return prompts[section][value]


async def aquestion_generator(categories: List[str], question_count: int = 10, difficulty: str = "Hard", context: str = None,
                             run_attempts=0, st_status=None, temperature=0) -> Dict[str, List[Dict[str, str]]]:
    """
    Uses an OpenAI model to generate a list of questions for each category.
    :param categories:
    :return:
    """
    llm = ChatOpenAI(temperature=temperature + float(run_attempts)/10, model_name='gpt-4')
    if context:
        context = "Here are some questions and answers that the user would like to be asked. \n```\n" + context + "\n```"
    else:
        context = ""
    system_prompt = get_prompt('question_generation', 'system_prompt')
    human_prompt = get_prompt('question_generation', 'human_prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    async def generate_for_category(category):
        try:
            st_status.update_status('Generating questions...') if st_status else None
            result = await llm_chain.arun(categories=[category], question_count=question_count, difficulty=difficulty,
                                          context=context, tags=['question_generation'], verbose=True)
            return eval(result)
        except Exception as e:
            if run_attempts > 10:
                raise e
            else:
                return await aquestion_generator(categories=[category], question_count=question_count, difficulty=difficulty,
                                                 run_attempts=run_attempts + 1)
        except:
            result = await llm.apredict(
                f"turn this into valid JSON so that your response can be parsed with python's `eval` function. {result}. This is a last resort, do NOT include any commentary or any warnings or anything else in this response. There should be no newlines or anything else. JUST the JSON.")
            return eval(result.split('```python')[1].split("```")[0].strip())

    # Execute for all categories in parallel
    results = await asyncio.gather(*(generate_for_category(cat) for cat in categories))

    # Combine results for all categories
    return {cat: res for cat, res in zip(categories, results)}


def question_generator(categories: List[str], question_count: int = 10, difficulty: str = "Hard", context: str = None,
                       run_attempts=0, st_status=None) -> Dict[str, List[Dict[str, str]]]:
    """
    Uses an OpenAI model to generate a list of questions for each category.
    :param categories:
    :return:
    """
    llm = ChatOpenAI(temperature=float(run_attempts)/10, model_name='gpt-4')
    if context:
        context = "Here are some questions and answers that the user would like to be asked. \n```\n" + context + "\n```"
    else:
        context = ""
    system_prompt = get_prompt('question_generation', 'system_prompt')
    human_prompt = get_prompt('question_generation', 'human_prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    try:
        st_status.update_status('Generating questions...') if st_status else None
        result = llm_chain.run(categories=categories, question_count=question_count, difficulty=difficulty,
                               context=context, verbose=True, tags=['question_generation'])
    except Exception as e:
        if run_attempts > 10:
            raise e
        else:
            return question_generator(categories=categories, question_count=question_count, difficulty=difficulty,
                                      run_attempts=run_attempts + 1)
    try:
        return eval(result)
    except:
        result = llm.predict(
            f"turn this into valid JSON so that your response can be parsed with python's `eval` function. {result}. This is a last resort, do NOT include any commentary or any warnings or anything else in this response. There should be no newlines or anything else. JUST the JSON.")
        return eval(result.split('```python')[1].split("```")[0].strip())


def fact_check_question(question, answer, category, try_attempts=0):
    """
    fact check a question, answer pair
    :param question:
    :param answer:
    :param category:
    :return:
    """
    search = BingSearchAPIWrapper(k=10)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Wikipedia",
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
            description='Usefil for when you want to use keywords to pull up the wikipedia page for a topic.'
        )
    ]

    llm = ChatOpenAI(temperature=float(try_attempts)/10, model_name='gpt-3.5-turbo')
    system_message = SystemMessage(content=get_prompt('fact_checking', 'system_prompt'))
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    human_prompt = get_prompt('fact_checking', 'human_prompt')
    human_prompt = human_prompt.format(question=question, answer=answer, category=category)
    parser = PydanticOutputParser(pydantic_object=FactCheckQuestion)
    fact_check_question_schema = {
        "title": "Fact Check Question",
        "description": "A model representing a trivia question with a user-provided answer, its category, and a fact check.",
        "type": "object",
        "properties": {
            "question": {
                "title": "Question",
                "description": "The trivia question",
                "type": "string"
            },
            "answer": {
                "title": "User's Answer",
                "description": "Answer to the trivia question provided by the user",
                "type": "string"
            },
            "category": {
                "title": "Category",
                "description": "Category of the question provided by the user",
                "type": "string"
            },
            "fact_check": {
                "title": "Fact Check Status",
                "description": "Whether the answer is correct or not",
                "type": "boolean"
            },
            "explanation": {
                "title": "Explanation",
                "description": "Comment on the answer provided by the user",
                "type": "string"
            }
        },
        "required": ["question", "answer", "category", "fact_check", "explanation"]
    }
    try:
        # result = llm_chain.run(question=question, answer=answer, category=category)
        result = agent_executor.run(input=human_prompt, question=question, answer=answer, category=category,
                                    verbose=True, tags=['fact_checking'])
        # Attempt to directly evaluate the result
        result = result.replace('"fact_check": false,', '"fact_check": False,').replace('"fact_check": true,',
                                                                                        '"fact_check": True,')

        try:
            result = eval(result)
            result['question'] = question
            return result
        except Exception as first_e:
            print(first_e)

        # Attempt to parse after trimming the result
        try:
            trimmed_result = result.split('}')[0] + '}'
            trimmed_result = eval(trimmed_result)
            trimmed_result['question'] = question
            return trimmed_result
        except Exception as second_e:
            print(second_e)

        # If both above attempts fail, use the LLM chain for structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt('fact_checking', 'system_prompt')),
            ("human", 'format the result as JSON {result}')
        ]
        )
        llm_chain = create_structured_output_chain(output_schema=fact_check_question_schema, llm=llm, prompt=prompt, verbose=True)
        result = llm_chain.run(result=result + '\nThe user submitted: ' + answer, question=question, answer=answer, category=category, tags=['fact_checking'])
        result['question'] = question
        return result

    except Exception as e:
        if try_attempts > 10:
            raise e
        else:
            return fact_check_question(question, answer, category, try_attempts=try_attempts + 1)


async def async_fact_check(question, answer, category):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, fact_check_question, question, answer, category)
    return result


def _fix_question(question, answer, category, explanation, previous_questions, run_attempts=0):
    """
    attempts to fix the question and answer pair
    :param question:
    :param answer:
    :param category:
    :param explanation:
    :param run_attempts:
    :return:
    """
    llm = ChatOpenAI(temperature=float(run_attempts)/10, model_name='gpt-4')
    system_prompt = get_prompt('question_fixing', 'system_prompt')
    human_prompt = get_prompt('question_fixing', 'human_prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = llm_chain.run(question=question, answer=answer, category=category, explanation=explanation,
                               previous_questions=previous_questions, verbose=True, tags=['question_fixing'])
    except Exception as e:
        if run_attempts > 10:
            raise e
        else:
            return _fix_question(question, answer, category, explanation, previous_questions,
                                 run_attempts=run_attempts + 1)
    result = result.replace('"fact_check": false,', '"fact_check": False,').replace('"fact_check": true,', '"fact_check": True,')
    try:
        return eval(result)
    except:
        result = llm.predict(
            f"turn this into valid JSON so that your response can be parsed with python's `eval` function. {result}. This is a last resort, do NOT include any commentary or any warnings or anything else in this response. There should be no newlines or anything else. JUST the JSON.")
        return eval(result.split('```python')[1].split("```")[0].strip())


def fix_question(question, answer, category, explanation, previous_questions, run_attempts=0):
    val = False
    k = 0
    while not val and k < 10:
        new_question = _fix_question(question, answer, category, explanation, previous_questions,
                                     run_attempts=run_attempts)
        # now
        new_fact_check = fact_check_question(new_question['question'], new_question['answer'], new_question['category'])
        val = new_fact_check['fact_check']
        question = new_fact_check['question']
        answer = new_fact_check['answer']
        category = category
        explanation = new_fact_check['explanation']
        k += 1

    return new_fact_check

async def async_fix_question(question, answer, category, explanation, previous_questions, run_attempts=0):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _fix_question, question, answer, category, explanation, previous_questions, run_attempts)
    return result


async def async_fix_and_check_question(question, answer, category, explanation, previous_questions, run_attempts=0):
    val = False
    k = 0
    while not val and k < 10:
        new_question = await async_fix_question(question, answer, category, explanation, previous_questions, run_attempts=run_attempts)
        new_fact_check = await async_fact_check(new_question['question'], new_question['answer'], new_question['category'])
        val = new_fact_check['fact_check']
        question = new_fact_check['question']
        answer = new_fact_check['answer']
        category = new_fact_check['category']
        explanation = new_fact_check['explanation']
        k += 1
    return new_fact_check


def grade_responses(response_json):
    """
    grades the responses from the question generator
    :param response_json:
    :return:
    """
    final_score = pd.DataFrame(columns=['question', 'answer', 'category', 'user_answer', 'grade'])
    total_score = pd.DataFrame(
        columns=['category', 'total_questions', 'total_correct', 'total_incorrect', 'total_score'])
    for key, val in eval(response_json).items():
        question, answer, category = key.split(' || ')
        user_answer = val['value']
        grade = _grade_answer(question, answer, user_answer)
        final_score.loc[len(final_score)] = [question, answer, category, user_answer, grade['grade']]

    for category in final_score.category.unique():
        category_score = final_score.loc[final_score.category == category]
        total_questions = len(category_score)
        total_correct = len(category_score.loc[category_score.grade == True])
        total_incorrect = len(category_score.loc[category_score.grade == False])
        total_score.loc[len(total_score)] = [category, total_questions, total_correct, total_incorrect,
                                             total_correct / total_questions]
    total_score.loc[len(total_score)] = ['total', len(final_score), len(final_score.loc[final_score.grade == True]),
                                         len(final_score.loc[final_score.grade == False]),
                                         len(final_score.loc[final_score.grade == True]) / len(final_score)]

    # make a table of the results
    return total_score, final_score


def _grade_answer(question, answer, user_answer, try_attempts=0):
    """
    the LLM checker that returns True if the answer is correct and False if its not
    :param question:
    :param answer:
    :param user_answer:
    :return:
    """
    llm = ChatOpenAI(temperature=float(try_attempts)/10, model_name='gpt-4')
    system_prompt = get_prompt('answer_grading', 'system_prompt')
    human_prompt = get_prompt('answer_grading', 'human_prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    )
    grade_answer_schema = {
        "title": "Grade Answer Output",
        "description": "Explanation of the grading result followed by the result itself.",
        "type": "object",
        "properties": {
            "explanation": {
                "title": "Explanation",
                "description": "A detailed explanation of why the user's answer is correct or incorrect.",
                "type": "string"
            },
            "grade": {
                "title": "Grade",
                "description": "Indicates if the user's answer is correct.",
                "type": "boolean"
            }
        },
        "required": ["explanation", "grade"]
    }
    llm_chain = create_structured_output_chain(output_schema=grade_answer_schema, llm=llm, prompt=prompt, verbose=True)
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = llm_chain.run(question=question, answer=answer, user_answer=user_answer, verbose=True, tags=['answer_grading'])
        return result
    except Exception as e:
        if try_attempts > 5:
            print(f'failed to grade answer: {user_answer} with error: {e}')
            return {
                'grade': True,
                'explanation': str(e)}  # This cannot fail, so if it does, just return True
        return _grade_answer(question, answer, user_answer, try_attempts=try_attempts + 1)


if __name__ == '__main__':
    categories = ['science', 'history', 'geography']
    result = question_generator(categories=categories)
    print(result)

    question = "Who discovered America?"
    answer = "Christopher Columbus"
    category = "geography"
    result = fact_check_question(question, answer, category)
    print(result)

    categories = ['science', 'history', 'geography']
    question_count = 10
    difficulty = 'Hard'
    context = pd.DataFrame(columns=['question', 'answer', 'category', 'difficulty'])
    context.loc[0] = ['Who discovered America?', 'Christopher Columbus', 'geography', 'Hard']
    context.loc[1] = ['Who was the first president of the United States?', 'George Washington', 'history', 'Hard']
    context.loc[2] = ['What is the capital of California?', 'Sacramento', 'geography', 'Hard']
    result = question_generator(categories=categories, question_count=question_count, difficulty=difficulty,
                                context=context.to_markdown())

    question = "Who is the current Prime Minister of the UK?"
    answer = "Boris Johnson"
    category = "news"
    explanation = "The current Prime Minister of the UK is Rishi Sunak, not Boris Johnson. Boris Johnson served as Prime Minister from 2019 to 2022."
    result = fix_question(question, answer, category, explanation, previous_questions=['who is the current president of the US?'])
    print(result)

    question = "Who is the current Prime Minister of the UK?"
    answer = "Boris Johnson"
    category = "news"
    explanation = "The current Prime Minister of the UK is not Boris Johnson."
    result = fix_question(question, answer, category, explanation,
                          previous_questions=['who is the current president of the US?'])
    print(result)
