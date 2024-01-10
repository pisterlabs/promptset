import importlib as __importlib
import json as __json

import cohere as __cohere
from openai import OpenAI as __OpenAI

from lib.configs import COHERE_API_KEY as __COHERE_API_KEY
from lib.configs import OPENAI_API_KEY as __OPENAI_API_KEY
from lib.configs import PALM_API_KEY as __PALM_API_KEY
from lib.types import Evaluation, Question

__all__ = [
    "OpenAIQuestionGeneratorAgent",
    "PalmQuestionGeneratorAgent",
    "CohereQuestionGeneratorAgent",
    "OpenAIResponseEvaluationAgent",
    "PalmResponseEvaluationAgent",
    "CohereResponseEvaluationAgent",
]


class __BaseAgent:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class OpenAIQuestionGeneratorAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __OpenAI(api_key=__OPENAI_API_KEY)
        self.system_prompt = """You are a non-technical interviewer that interviews \
across the following categories:
- personal
- role-specific
- behavioural
- situational

You will be provided with a candidate's description.

Generate {n_questions} questions, ensuring that there is a question for each category \
and the questions should be based on the candidate's description.

* You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
and don't include the markdown syntax anywhere.

JSON format:
[
    {{"question": "<personal_question>", "type": "personal"}},
    {{"question": "<role_specific_question>", "type": "role-specific"}},
    {{"question": "<behavioural_question>", "type": "behavioural"}},
    {{"question": "<situational_question>", "type": "situational"}},
    ...more questions to make up {n_questions} questions
]"""

        self.user_prompt = "Candidate Description:\n{description}"

    def __call__(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def run(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def _generate(self, description: str, n_questions: int) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int): The number of questions to generate.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        try:
            # Ensure that there are at least 4 questions
            if n_questions < 4:
                n_questions = 4

            output = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(n_questions=n_questions),
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(description=description),
                    },
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            questions = __json.loads(output.choices[0].message.content or "[]")

            return questions
        except Exception:
            return None


class PalmQuestionGeneratorAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __importlib.import_module("google.generativeai")
        self.client.configure(api_key=__PALM_API_KEY)
        self.system_prompt = """You are a non-technical interviewer that interviews \
across the following categories:
- personal
- role-specific
- behavioural
- situational

You will be provided with a candidate's description.

Generate {n_questions} questions, ensuring that there is a question for each category \
and the questions should be based on the candidate's description.

* You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
and don't include the markdown syntax anywhere.

JSON format:
[
    {{"question": "<personal_question>", "type": "personal"}},
    {{"question": "<role_specific_question>", "type": "role-specific"}},
    {{"question": "<behavioural_question>", "type": "behavioural"}},
    {{"question": "<situational_question>", "type": "situational"}},
    ...more questions to make up {n_questions} questions
]


===
Candidate Description:
{description}"""

    def __call__(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def run(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def _generate(self, description: str, n_questions: int) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int): The number of questions to generate.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        try:
            # Ensure that there are at least 4 questions
            if n_questions < 4:
                n_questions = 4

            output = self.client.generate_text(
                model="models/text-bison-001",
                prompt=self.system_prompt.format(n_questions=n_questions, description=description),
                temperature=1,
                max_output_tokens=1024,
            )
            questions = __json.loads(output.result or "[]")

            return questions
        except Exception:
            return None


class CohereQuestionGeneratorAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __cohere.Client(__COHERE_API_KEY)
        self.system_prompt = """You are a non-technical interviewer that interviews \
across the following categories:
- personal
- role-specific
- behavioural
- situational

You will be provided with a candidate's description.

Generate {n_questions} questions, ensuring that there is a question for each category \
and the questions should be based on the candidate's description.

* You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
and don't include the markdown syntax anywhere.

JSON format:
[
    {{"question": "<personal_question>", "type": "personal"}},
    {{"question": "<role_specific_question>", "type": "role-specific"}},
    {{"question": "<behavioural_question>", "type": "behavioural"}},
    {{"question": "<situational_question>", "type": "situational"}},
    ...more questions to make up {n_questions} questions
]


===
Candidate Description:
{description}"""

    def __call__(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def run(self, description: str, n_questions: int = 4) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int, optional): The number of questions to generate. Defaults to 4.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        # Generate questions
        questions = self._generate(description, n_questions)

        return questions

    def _generate(self, description: str, n_questions: int) -> list[Question] | None:
        """
        Generate interview questions based on the given description.

        Args:
            description (str): The description used as input for question generation.
            n_questions (int): The number of questions to generate.

        Returns:
            list[Question] | None: A list of generated interview questions or None if an error occurs.
        """

        try:
            # Ensure that there are at least 4 questions
            if n_questions < 4:
                n_questions = 4

            output = self.client.generate(
                model="command",
                prompt=self.system_prompt.format(n_questions=n_questions, description=description),
                temperature=1,
                max_tokens=1024,
            )
            questions = __json.loads(output.generations[0].text or "[]")

            return questions
        except Exception:
            return None


class OpenAIResponseEvaluationAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __OpenAI(api_key=__OPENAI_API_KEY)
        self.system_prompt = """You are an interviewer evaluating a candidate's \
response to an interview question. Your task is to:
- Evaluate the candidate's response on the scale of "good", "average", and "bad".
- Provide a reason for why it's categorized as good, average, or bad.
- Offer constructive feedback or suggestions for improvement.
- Provide 2 samples of good responses.

You will be provided with an interview question and a candidate response.

Evaluate and provide output in the following JSON format:
{{
    "evaluation": "good, average, or bad",
    "reason": "Reason why it's good, average, or bad",
    "feedback": "Feedback or suggestions for improvement",
    "samples": [
        "<Good response 1>",
        "<Good response 2>"
    ]
}}"""
        self.user_prompt = """QUESTION:
{question}

RESPONSE:
{response}"""

    def __call__(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def run(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def _generate(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        try:
            output = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(question=question, response=response),
                    },
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            questions = __json.loads(output.choices[0].message.content or "{}")

            return questions
        except Exception:
            return None


class PalmResponseEvaluationAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __importlib.import_module("google.generativeai")
        self.client.configure(api_key=__PALM_API_KEY)
        self.system_prompt = """You are an interviewer evaluating a candidate's \
response to an interview question. Your task is to:
- Evaluate the candidate's response on the scale of "good", "average", and "bad".
- Provide a reason for why it's categorized as good, average, or bad.
- Offer constructive feedback or suggestions for improvement.
- Provide 2 samples of good responses.

You will be provided with an interview question and a candidate response.

Evaluate and provide output in the following JSON format:
{{
    "evaluation": "good, average, or bad",
    "reason": "Reason why it's good, average, or bad",
    "feedback": "Feedback or suggestions for improvement",
    "samples": [
        "Good response 1",
        "Good response 2"
    ]
}}


===
QUESTION:
{question}

RESPONSE:
{response}"""

    def __call__(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def run(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def _generate(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        try:
            output = self.client.generate_text(
                model="models/text-bison-001",
                prompt=self.system_prompt.format(question=question, response=response),
                temperature=1,
                max_output_tokens=1024,
            )

            evaluations = __json.loads(output.result)

            return evaluations
        except Exception:
            return None


class CohereResponseEvaluationAgent(__BaseAgent):
    def __init__(self):
        super().__init__()

        self.client = __cohere.Client(__COHERE_API_KEY)
        self.system_prompt = """You are an interviewer evaluating a candidate's \
response to an interview question. Your task is to:
- Evaluate the candidate's response on the scale of "good", "average", and "bad".
- Provide a reason for why it's categorized as good, average, or bad.
- Offer constructive feedback or suggestions for improvement.
- Provide 2 samples of good responses.

You will be provided with an interview question and a candidate response.

Evaluate and provide output in the following JSON format:
{{
    "evaluation": "good, average, or bad",
    "reason": "Reason why it's good, average, or bad",
    "feedback": "Feedback or suggestions for improvement",
    "samples": [
        "Good response 1",
        "Good response 2"
    ]
}}


===
QUESTION:
{question}

RESPONSE:
{response}"""

    def __call__(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def run(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        # Generate questions
        evaluation = self._generate(question, response)

        return evaluation

    def _generate(self, question: str, response: str) -> Evaluation | None:
        """
        Evaluate a candidate's response to an interview question.

        Args:
            question (str): The interview question.
            response (str): The candidate's response.

        Returns:
            Evaluation | None: The evaluation of the candidate's response or None if an error occurred.
        """

        try:
            output = self.client.generate(
                model="command",
                prompt=self.system_prompt.format(question=question, response=response),
                temperature=1,
                max_tokens=1024,
            )

            evaluations = __json.loads(output.generations[0].text)

            return evaluations
        except Exception:
            return None
