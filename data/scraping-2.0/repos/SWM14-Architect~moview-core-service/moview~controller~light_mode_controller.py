from flask import make_response, jsonify, request, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restx import Resource, Namespace
from http import HTTPStatus

from moview.config.container.container_config import ContainerConfig
from moview.config.loggers.mongo_logger import *
from moview.decorator.timing_decorator import api_timing_decorator
from moview.decorator.validation_decorator import validate_char_count
from moview.controller.constants.input_data_constants import (MAX_COMPANY_NAME_LENGTH, MAX_POSITION_NAME_LENGTH,
                                                              MAX_KEYWORD_LENGTH)
from moview.exception.retry_execution_error import RetryExecutionError
import openai

api = Namespace('light_mode', description='light mode api')


@api.route('/light')
class LightModeConstructor(Resource):

    @api_timing_decorator
    @validate_char_count({
        'company_name': MAX_COMPANY_NAME_LENGTH,
        'job_group': MAX_POSITION_NAME_LENGTH,
        'keyword': MAX_KEYWORD_LENGTH
    })
    @jwt_required()
    def post(self):
        user_id = str(get_jwt_identity())
        g.user_id = user_id
        request_body = request.get_json()

        interviewee_name = request_body['interviewee_name']
        company_name = request_body['company_name']
        job_group = request_body['job_group']
        keyword = request_body['keyword']

        interview_service = ContainerConfig().interview_service
        light_mode_service = ContainerConfig().light_mode_service

        # 1. Interview Document 생성
        try:
            interview_document_id = interview_service.create_interview(
                user_id=user_id,
            )
            g.interview_id = interview_document_id

        except Exception as e:
            error_logger(msg="CREATE INTERVIEW DOCUMENT ERROR", error=e)
            raise e

        # 2. Initial Question 생성
        try:
            result = light_mode_service.ask_light_question_to_interviewee(
                interviewee_name=interviewee_name,
                company_name=company_name,
                job_group=job_group, keyword=keyword)

        except RetryExecutionError as e:
            error_logger(msg="RETRY EXECUTION ERROR")
            raise e

        except openai.error.RateLimitError as e:
            error_logger(msg="RATE LIMIT ERROR")
            raise e

        except Exception as e:
            error_logger(msg="CREATE INTERVIEW DOCUMENT ERROR", error=e)
            raise e

        # Parse Error 발생했을 경우 500 에러 반환
        if result is None:
            return make_response(jsonify(
                {'message': {
                    'error': 'Oops! 당신의 질문이 우주로 떠나버렸어! 다시 시도해주세요.',
                    'error_message': 'Parse Error'
                }}
            ), HTTPStatus.INTERNAL_SERVER_ERROR)

        # 3. Interview Document에 Initial Input Data Document ID 업데이트
        try:
            interview_service.update_interview_with_initial_input_data(
                user_id=user_id,
                interview_document_id=interview_document_id,
                input_data_document_id=result['input_data_document']
            )
        except Exception as e:
            error_logger(msg="CREATE INTERVIEW DOCUMENT ERROR", error=e)
            raise e

        execution_trace_logger("LIGHT MODE CONTROLLER: POST",
                               interviewee_name=interviewee_name,
                               company_name=company_name,
                               job_group=job_group,
                               keyword=keyword)

        return make_response(jsonify(
            {'message': {
                'light_questions': [{"question_id": str(object_id), "content": question} for object_id, question in
                                    result['question_document_list']],
                'interview_id': interview_document_id
            }}
        ), HTTPStatus.OK)
