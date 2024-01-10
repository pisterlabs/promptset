# from model.employee import Employee
import openai
import json
from decouple import config

from data_access.profiling.repository import employee_repository
from data_access.reporting.repository import reporting_repository
from service.employee_service import EmployeeService
from model.employee import Employee
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

class ReportingService: 

    def __init__(self):
        self._open_Api_key = config('OPEN_API_KEY')

    def generate_report(self, employee_id):
        employee_json = employee_repository.get_employee_details(employee_id)
        employee = Employee.from_json(employee_json)
        feedbacks = employee_repository.get_employee_feedbacks(employee_id)
        employee_name = employee.name
        feedbacks = [feedback_object for feedback_object in feedbacks['feedbacks']]

        report_json = self.get_report_from_feedbacks(employee_name, feedbacks)
        pdf_buffer = reporting_repository.generate_pdf(report_json)
        return pdf_buffer

    def get_report_from_feedbacks(self, employee_name, feedbacks):

        llm_model = "gpt-3.5-turbo"

        system_message = SystemMessage(content=(
            """
            Analyze the provided feedback to create a comprehensive report on an employee's performance. Pay particular attention to the following aspects:

            **Strengths:**

            * Identify specific examples and evidence from the feedback that highlight the employee's strengths.
            * Present these points in a clear and concise manner, avoiding repetition of the exact wording from the feedback itself.

            **Weaknesses:**

            * Identify areas where the employee can improve based on the feedback.
            * Provide constructive criticism and specific examples to support your analysis.
            * Avoid using accusatory language and focus on offering actionable suggestions for improvement.

            **Areas for Improvement:**

            * Go beyond simply listing weaknesses and offer concrete recommendations for development.
            * Align your suggestions with the identified areas for improvement and provide clear steps for the employee to take.
            * Prioritize the recommendations based on their impact and feasibility.

            **Additional Points:**

            * Keep the report concise and focused on key takeaways.
            * Use clear and concise language to communicate your findings effectively.
            * Avoid subjective statements and focus on objective analysis of the feedback.
            * Remember to maintain a neutral and unbiased tone throughout the report.
            * **Note:** Don't use the exact wording of the feedback in the output itself. Analyze and summarize the information to provide a comprehensive and insightful report.

            **Safety Guidelines:**

            * This report should not be harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
            * It should not be insensitive, sexist, racist, or socially inappropriate.
            * Avoid controversial or objectionable content based on common sense ethical and moral standards.
            * Do not promote violence, hatred, or discrimination.
            * The report should not be sexually suggestive in nature.
            * Do not seek private information about individuals.

            **Please respond creatively within these guidelines.**
            **Please ensure your report adheres to all ethical and safety guidelines.**
            """)
        )

        response_schemas = [
            ResponseSchema(
                name="response",
                description="""object contains name, strengths, weaknesses and areas for improvement in the following format:
                  {{ "employee_name": string // The name of the employee, 
                  "strengths": string // A string separated by new lines of the employee's identified strengths, supported by specific examples from the feedback., 
                  "weaknesses": string // A string separated by new lines of areas where the employee can improve, with constructive suggestions and clear paths for development., 
                  "areas_for_improvement": string // A string separated by new lines of specific and actionable recommendations for addressing the identified weaknesses, prioritized based on their impact and feasibility. }}
                ]
                """,
            )
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions().replace(
            '"response": string', '"response": object'
        )
        human_message_prompt_template = HumanMessagePromptTemplate.from_template(
            """
            **Employee Name:**
            {employee_name}

            **Feedback:**
            {feedback_list}

            **format_instructions**
            {format_instructions}
            """
        )

        feedbacks_str = '\n'.join(feedbacks)

        chat_prompt_template = ChatPromptTemplate.from_messages([system_message, human_message_prompt_template])
        formatted_user_input = chat_prompt_template.format_messages(
            employee_name=employee_name, feedback_list=feedbacks_str, format_instructions=format_instructions
        )

        chat = ChatOpenAI(temperature=0.0, model=llm_model, openai_api_key=self._open_Api_key)
        response = chat(formatted_user_input)
        print(response)
        if response and response.content:
            output_dict = output_parser.parse(response.content)
            print(output_dict)
            print(output_dict.get('response'))
            return output_dict.get('response')
        else:
            return None
