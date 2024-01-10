import epicbox
import json
import openai
from fastapi import (
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session
from textwrap import dedent

import homeworks
from davinci.database import get_session
from davinci.settings import settings
from solutions import schemas
from solutions.schemas import SolutionError

openai.api_key = settings.openai_api_key


class SolutionService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session

    def extend_solution_text(self, solution_text: str) -> str:
        pre_solution_text = dedent("""
            import json
            input_text = input()
            params = json.loads(input_text)
        """)
        post_solution_text = dedent("""
            result = solution(*params.values())
            print(result)
        """)
        solution_text = ''.join([
            pre_solution_text,
            solution_text,
            post_solution_text
        ])

        return solution_text

    def check(
        self,
        homework_number: int,
        solution_text: str
    ) -> schemas.SolutionResponse:
        homework = (
            self.session
            .query(homeworks.models.Homework)
            .filter(
                homeworks.models.Homework.number == homework_number
            )
            .first()
        )

        if not homework:
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        if homework.is_function:
            solution_text = self.extend_solution_text(solution_text)

        # for test_sample in test_samples_set:
        for test_sample in homework.test_samples:
            epicbox.configure(
                profiles=[
                    epicbox.Profile('python', 'python:3.6.5-alpine')
                ]
            )

            files = [{'name': 'main.py', 'content': solution_text.encode()}]
            limits = {'cputime': 1, 'memory': 64}
            solution_result = epicbox.run(
                profile_name='python',
                command='python3 main.py',
                stdin=test_sample.input.encode(),
                files=files,
                limits=limits
            )

            actual_output = solution_result['stdout'].decode().strip()
            if actual_output != test_sample.expected:
                serialized_solution_result = json.dumps({
                    k: v.decode() if isinstance(v, bytes) else v
                    for k, v in solution_result.items()
                })

                error = SolutionError(
                    message=serialized_solution_result,
                    input=test_sample.input,
                    expected=test_sample.expected,
                    actual=actual_output,
                )

                result = schemas.SolutionResponseFail(
                    error=error
                )
                break
        else:
            result = schemas.SolutionResponseSuccess()

        return result


class AISolutionService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session

    def check(
        self,
        homework_number: int,
        solution_text: str,
    ) -> str:
        homework = (
            self.session
            .query(homeworks.models.Homework)
            .filter(
                homeworks.models.Homework.number == homework_number
            )
            .first()
        )

        if not homework:
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        response = openai.ChatCompletion.create(
            model="gpt-4-0314",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Ось завдання по програмуванню на Python:\n\n{homework.description}"},
                {"role": "assistant", "content": "Ok, got it."},
                {"role": "user", "content": f"Чи є наступний Python код коректним рішенням для цього завдання?:\n\n{solution_text}"}
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = response.choices[0].message.content

        return result


    def get_prompt(
        self,
        homework_number: int
    ) -> str:
        homework = (
            self.session
            .query(homeworks.models.Homework)
            .filter(
                homeworks.models.Homework.number == homework_number
            )
            .first()
        )

        if not homework:
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        response = openai.ChatCompletion.create(
            model="gpt-4-0314",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Ось завдання по програмуванню на Python:\n\n{homework.description}"},
                {"role": "assistant", "content": "Ok, got it."},
                {"role": "user", "content": f"Дай підказку як його виконати, але без остаточного рішення."}
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = response.choices[0].message.content

        return result
