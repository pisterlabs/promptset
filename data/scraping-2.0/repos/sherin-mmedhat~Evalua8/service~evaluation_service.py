from decouple import config
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

from data_access.profiling.repository import evaluation_repository
from data_access.profiling.repository import employee_repository
from model.evaluation import EvaluationList
class EvaluationService:

    def __init__(self):
        self._open_Api_key = config('OPEN_API_KEY')

    def get_evaluation(self, evaluator_id: int, employee_id: int):
        return evaluation_repository.get_evaluation(employee_id=employee_id, evaluator_id=evaluator_id)

    def score_evaluations(self, employee_id, evaluator_id):

        feedbacks = employee_repository.get_employee_feedbacks_by_evaluator(employee_id, evaluator_id)
        print(feedbacks)
        evaluations_data = evaluation_repository.get_evaluation(employee_id, evaluator_id)

        evaluationList = EvaluationList(**evaluations_data)
        questions_list = [evaluation.question for evaluation in evaluationList.evaluations]

        score_response = self.give_rates_for_questions(feedbacks, questions_list)

        print(score_response)
        for score_object in score_response:
            print(score_object["question"])
            print(score_object["score"])
            evaluation_repository.update_evaluation_score(employee_id, evaluator_id, score_object["question"],
                                                          score_object["score"])

        return self.get_evaluation(evaluator_id, employee_id)

    def give_rates_for_questions(self, feedbacks, questions):

        llm_model = "gpt-3.5-turbo"

        system_message = SystemMessage(content=(
            """
            You are now tasked with scoring an employee's performance on a scale of 1 to 10 (1 being the lowest and 10 being the highest) for each of the specified competencies. Your evaluation should be based solely on the information provided in the employee feedback.

            **It's important to remain objective and unbiased in your assessment. Focus only on the feedback provided and avoid making assumptions or inferences not directly supported by the text.**

            **Scoring Format:**

            1 - Unsatisfactory
            2 - Below Average
            3 - Average
            4 - Above Average
            5 - Neutral (Insufficient information to score)
            6 - Good
            7 - Very Good
            8 - Excellent
            9 - Outstanding
            10 - Exceptional

            Please provide a numerical rating for each competency. If you find that a competency cannot be adequately answered from the feedback, assign a neutral score of 5.

            **Remember:** Your goal is to provide a fair and accurate assessment based on the available information.
            """
        ))

        response_schemas = [
            ResponseSchema(
                name="response",
                description="""
                array contains question, score in the following format: [
                {{ "question": string // Each competency from the given competencies.',
                "score": int // number representing the score given for each competency according to feedback.', }}
            ]
            """,
            )
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions().replace(
            '"response": string', '"response": array of objects'
        )

        human_message_prompt_template = HumanMessagePromptTemplate.from_template(
            """
            **Feedback:**
            {feedback_list}

            **Competencies:**
            {question_list}

            **format_instructions**
            {format_instructions}
            """
        )

        feedbacks_str = ', '.join(feedbacks)
        questions_str = ', '.join(questions)

        chat_prompt_template = ChatPromptTemplate.from_messages([system_message, human_message_prompt_template])

        formatted_user_input = chat_prompt_template.format_messages(
            feedback_list=feedbacks_str, question_list=questions_str, format_instructions=format_instructions
        )

        chat = ChatOpenAI(temperature=0.0, model=llm_model, openai_api_key=self._open_Api_key)
        response = chat(formatted_user_input)

        if response and response.content:
            output_dict = output_parser.parse(response.content)
            print(output_dict)
            print(output_dict.get('response'))
            return output_dict.get('response')
        else:
            return None