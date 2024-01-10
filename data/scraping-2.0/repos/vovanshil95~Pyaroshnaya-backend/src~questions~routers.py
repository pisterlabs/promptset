import re
import uuid
from datetime import datetime
from typing import Callable
from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, and_, update, delete
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from auth.routes import get_access_token
from auth.utils import AccessTokenPayload
from payment.utils import Paywall, PaywallManager, PaywallManagerTest
from questions.models import Category as CategoryModel, Option, Prompt
from questions.models import Answer as AnswerModel
from questions.models import Question as QuestionModel
from questions.schemas import Question as QuestionSchema, GptAnswerResponse, NewAnswers
from questions.schemas import Category as CategorySchema
from questions.schemas import Answer as AnswerSchema
from questions.schemas import Option as OptionSchema
from questions.schemas import CategoriesResponse, QuestionsResponse
from history.models import GptInteraction
from database import get_async_session
from utils import BaseResponse, try_uuid
from config import OPENAI_API_KEY, PAYWALL_ON

router = APIRouter(prefix='/question',
                   tags=['Questions'])

QuestionsData = list[tuple[QuestionModel, list[str], list[str], list[uuid.UUID]]]

async def get_filled_prompt(questions: list[QuestionSchema],
                            prompt: list[str],
                            session: AsyncSession) -> str:
    option_ids = []

    for question in questions:
        if question.answer is None:
            continue
        if question.questionType == 'options' or (question.questionType == 'semi-options'
                                                  and isinstance(try_uuid(question.answer), uuid.UUID)):
            try:
                uuid.UUID(hex=question.answer)
            except ValueError:
                raise HTTPException(status_code=422, detail='optional questions must have uuid in answer')
            option_ids.append(question.answer)
        if question.questionType == 'numeric':
            if not question.answer.isnumeric():
                raise HTTPException(status_code=422, detail='answers to questions with numeric type must be numeric')

    if option_ids != []:
        options = (await session.execute(select(Option).where(Option.id.in_(option_ids)))).scalars().all()
        options_dict = {}
        for option in options:
            options_dict[option.id] = option.text_to_prompt
    answers = [None if question.answer is None
               else options_dict[uuid.UUID(hex=question.answer)]
               if question.questionType == 'options' or
                  (question.questionType == 'semi-options'
                   and isinstance(try_uuid(question.answer), uuid.UUID))
               else question.answer
               for question in questions]

    answers.insert(0, None)

    clean_prompt = deepcopy(prompt)

    try:
        for prompt_el in prompt:
            matches = [int(el) for el in re.findall(r'\{(\d+)\}', prompt_el)]
            if len(matches) > 0 and all([answers[match] is None for match in matches]):
                clean_prompt.remove(prompt_el)
    except IndexError:
        raise HTTPException(status_code=404, detail='Invalid prompt')

    filled_prompt = '\n'.join(clean_prompt).format(*answers)

    return filled_prompt

async def filled_prompt_generator():
    return get_filled_prompt

def get_gpt_send():
    async def get_gpt_response(questions: list[QuestionSchema],
                               prompt: list[str],
                               session: AsyncSession) -> str:

        filled_prompt = await get_filled_prompt(questions, prompt, session)

        max_time_wait = 300

        async with httpx.AsyncClient(timeout=httpx.Timeout(max_time_wait)) as client:
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                },
                json={
                    'model': 'gpt-4',
                    'messages': [{'role': 'user', 'content': filled_prompt}]
                }
            )

        return response.json()['choices'][0]['message']['content']

    return get_gpt_response

async def get_question_data(user_id: uuid.UUID, session: AsyncSession, category_id: uuid.UUID | None=None) -> QuestionsData:
    questions = (await session.execute(select(QuestionModel,
                                              func.array_agg(AnswerModel.text),
                                              func.array_agg(aggregate_order_by(Option.option_text, Option.option_text)),
                                              func.array_agg(AnswerModel.id),
                                              func.array_agg(aggregate_order_by(Option.id, Option.option_text)))
                                       .join(AnswerModel)
                                       .join(Option, isouter=True)
                                       .where(and_(QuestionModel.category_id==category_id if category_id is not None else True,
                                                   AnswerModel.interaction_id.is_(None),
                                                   AnswerModel.template_id.is_(None),
                                                   AnswerModel.user_id == user_id))
                                       .group_by(QuestionModel.id)
                                       .order_by(QuestionModel.order_index))).all()
    questions = list(map(lambda q: (q[0],
                                    list(zip(*list(set(list(zip(q[1], q[3]))))))[0] if q[0].type_ == 'options' else q[1],
                                    list(filter(lambda opt: opt is not None, q[2])),
                                    list(zip(*list(set(list(zip(q[1], q[3]))))))[1] if q[0].type_ == 'options' else q[3],
                                    list(filter(lambda opt: opt is not None, q[4]))), questions))
    return questions

def get_question_schemas(questions: QuestionsData) -> list[QuestionSchema]:
    return list(map(lambda question_data:
                    QuestionSchema(id=question_data[0].id,
                                   question=question_data[0].question_text,
                                   snippet=question_data[0].snippet,
                                   options=[OptionSchema(id=id, text=text) for id, text in zip(question_data[4], question_data[2])]
                                   if len(question_data[2]) != 0 else None,
                                   isRequired=question_data[0].is_required,
                                   categoryId=question_data[0].category_id,
                                   answer=question_data[1][0]
                                   if len(question_data[1]) == 1 else None,
                                   answers=question_data[1]
                                   if len(question_data[1]) > 1 else None,
                                   questionType=question_data[0].type_), questions))


@router.get('/categories',
            dependencies=[Depends(get_access_token)],
            responses={200: {'model': CategoriesResponse},
                       400: {'model': BaseResponse}})
async def get_categories(session: AsyncSession = Depends(get_async_session)) -> CategoriesResponse:
    return CategoriesResponse(message='status success',
                              categories=list(map(lambda category_model:
               CategorySchema(id=category_model.id,
                              title=category_model.title,
                              description=category_model.description,
                              parentId=category_model.parent_id,
                              isMainScreenPresented=category_model.is_main_screen_presented,
                              isCategoryScreenPresented=category_model.is_category_screen_presented,
                              orderIndex=category_model.order_index
                              ),
               (await session.execute(select(CategoryModel)
                                      .order_by(CategoryModel.order_index))).scalars().all())))

@router.get('/questions', responses={200: {'model': QuestionsResponse},
                                          400: {'model': BaseResponse},
                                          401: {'model': BaseResponse, 'description': 'User is not authorized'},
                                          498: {'model': BaseResponse, 'description': 'the access token is invalid'}})
async def get_questions(categoryId: uuid.UUID | None=None,
                        user_token: AccessTokenPayload=Depends(get_access_token),
                        session: AsyncSession=Depends(get_async_session)) -> QuestionsResponse:
    return QuestionsResponse(message='status success',
                             questions=get_question_schemas(await get_question_data(user_token.id,
                                                                                    session,
                                                                                    categoryId)))

@router.post('/response',
             responses={200: {'model': GptAnswerResponse},
                        400: {'model': BaseResponse, 'description': 'required fields not filled'},
                        401: {'model': BaseResponse, 'description': 'User is not authorized'}})
async def gpt_response(paywall_manager: Paywall=Depends(PaywallManager() if PAYWALL_ON else PaywallManagerTest()),
                       user_token: AccessTokenPayload=Depends(get_access_token),
                       session: AsyncSession=Depends(get_async_session),
                       gpt_send: Callable=Depends(get_gpt_send)) -> GptAnswerResponse:
    questions_data = await get_question_data(user_token.id, session, paywall_manager.category_id)
    questions = get_question_schemas(questions_data)
    for question in questions:
        if question.isRequired and ((not question.answers and not question.answer) or question.answer == ''):
            raise HTTPException(status_code=400, detail='required fields not filled')

    prompt = (await session.execute(select(Prompt.text)
                                    .where(Prompt.category_id == paywall_manager.category_id)
                                    .order_by(Prompt.order_index))).scalars().all()

    response = await gpt_send(questions, prompt, session)

    paywall_manager.symbols_in_response = len(response)

    interaction_id = uuid.uuid4()
    answer_ids = []
    new_answers = []

    session.add(GptInteraction(id=interaction_id,
                               time_happened=(interaction_time:=datetime.now()),
                               response=response))

    for question_data in questions_data:
        answer_ids.extend(question_data[3])
        for ans_text in question_data[1]:
            new_answers.append(AnswerModel(id=uuid.uuid4(),
                                           question_id=question_data[0].id,
                                           text=ans_text,
                                           user_id=user_token.id))

    await session.flush()
    await session.execute(update(AnswerModel).where(AnswerModel.id.in_(answer_ids)).values(interaction_id=interaction_id))
    session.add_all(new_answers)
    return GptAnswerResponse(answerId=interaction_id,
                             dateTime=interaction_time,
                             message='status success',
                             questions=questions,
                             gptResponse=response)

@router.post('/questions', responses={200: {'model': QuestionsResponse},
                                           401: {'model': BaseResponse, 'description': 'User is not authorized'},
                                           404: {'model': BaseResponse, 'description': 'Question with this id doesnt exist'},
                                           422: {'model': BaseResponse, 'description': 'Validation error: answer or answers must be specified'},
                                           498: {'model': BaseResponse, 'description': 'the access token is invalid'}})
async def answer(answer: AnswerSchema,
                 user_token: AccessTokenPayload=Depends(get_access_token),
                 session: AsyncSession=Depends(get_async_session)) -> QuestionsResponse:

    if (question := await session.get(QuestionModel, answer.questionId)) is None:
        raise HTTPException(404, 'Question with this id doesnt exist')

    if answer.answers is None:
        if answer.answer is not None:
            if question.type_ == 'numeric' and not answer.answer.isnumeric():
                raise HTTPException(status_code=422, detail='answers to questions with numeric type must be numeric')
            if question.type_ == 'options':
                try:
                    uuid.UUID(hex=answer.answer)
                except ValueError:
                    raise HTTPException(status_code=422, detail='optional questions must have uuid in answer')
        answer_model = (await session.execute(
                select(AnswerModel)
                .where(and_(AnswerModel.question_id == answer.questionId,
                            AnswerModel.user_id == user_token.id,
                            AnswerModel.template_id.is_(None),
                            AnswerModel.interaction_id.is_(None)))
        )).scalars().first()
        answer_model.text = answer.answer
    else:
        await session.execute(delete(AnswerModel).where(and_(
            AnswerModel.question_id == answer.questionId,
            AnswerModel.user_id == user_token.id,
            AnswerModel.template_id.is_(None),
            AnswerModel.interaction_id.is_(None)
        )))
        session.add_all(list(map(lambda a:
                                 AnswerModel(id=uuid.uuid4(),
                                             question_id=answer.questionId,
                                             text=a,
                                             user_id=user_token.id),
                                 answer.answers)))
    await session.commit()
    return QuestionsResponse(message='status success',
                             questions=get_question_schemas(
                                 await get_question_data(user_id=user_token.id,
                                                         session=session,
                                                         category_id=question.category_id)))

@router.post('/allQuestions')
async def answer_all_questions(new_answers: NewAnswers,
                               user_token: AccessTokenPayload=Depends(get_access_token),
                               session: AsyncSession=Depends(get_async_session)) -> QuestionsResponse:

    question_ids = [answer.quetionId for answer in new_answers.answers]
    answer_texts = {answer.quetionId: answer.answer for answer in new_answers.answers}

    answers = (await session.execute(
        select(AnswerModel)
        .join(QuestionModel)
        .join(CategoryModel)
        .where(and_(CategoryModel.id == new_answers.categoryId,
                    AnswerModel.user_id == user_token.id,
                    AnswerModel.question_id.in_(question_ids),
                    AnswerModel.interaction_id.is_(None),
                    AnswerModel.template_id.is_(None))))
               ).scalars().all()

    for answer in answers:
        answer.text = answer_texts[answer.question_id]

    await session.commit()

    return QuestionsResponse(message='status success',
                             questions=get_question_schemas(
                                 await get_question_data(user_id=user_token.id,
                                                         session=session,
                                                         category_id=new_answers.categoryId)))
