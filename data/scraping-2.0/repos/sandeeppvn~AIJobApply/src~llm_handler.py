import logging
from typing import Dict

import pandas as pd
from gptrim import trim
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Configure logging for the application.
logging.basicConfig(level=logging.INFO)

class CustomJobApplicationMaterials(BaseModel):
    """
    Model for custom job application materials.
    """
    cover_letter: str = Field(description="Customized Cover Letter")
    resume_summary: str = Field(description="Enhanced Resume Summary")
    missing_keywords: str = Field(description="Missing Keywords")
    email_content: str = Field(description="Refined Email Content")
    email_subject: str = Field(description="Email Subject Line")
    linkedin_note: str = Field(description="LinkedIn Note")

class LLMConnectorClass:
    """
    Connects with a Language Learning Model (LLM) to generate custom content.
    """
    prompt_template = """
        Prompt: Job Description Refinement and Application Materials Creation

        Task Overview: 
        - Start with analyzing the job description. 
        - Then, create specific application materials.
        Return only the final output json with all the keys and values populated.

        Step 1: Ana;lyze the Job Description
        - Input: Raw Job Description: {job_description}, Job Title: {position}
        - Sub-steps:
            1.1 Analyze the raw job description for key roles and responsibilities.
            1.2 Identify and list essential hard skills such as technical skills and tools.
            1.3 Identify soft skills like communication, teamwork, problem-solving.
            1.4 Understand the company's culture, values, mission, and vision.

        Step 2: Enhance the Resume
        - Reference the updated job description from Step 1.
        - Sub-steps:
            3.1 Utilize the Resume Template: ``` {resume_template} ``` and the Resume Professional Summary: ``` {resume_professional_summary} ```
            3.2 Revise the professional summary to align with the new job description. Have a statement "Seeking a {position} at {company_name} ..." in it and provide it in the "resume_summary" key.
            3.4 Provide the technical skills and tools that are missing in the resume but are required for the job (based on the job description). Provide only technical keywords which generally reflect hard skills.
            Provide the missing keywords in the "missing_keywords" key.
        - Aim: Reflect the key aspects of the job description accurately.
        - Place the outputs in the keys "resume_summary" and "missing_keywords" in the output JSON.
        
        Step 3: Craft a Customized Cover Letter
        - Use the updated job description from Step 1 and the resume from Step 2.
        - Sub-steps:
            2.1 Start with the Cover Letter Template: ``` {cover_letter_template} ```
            2.2 Integrate elements from the updated job description relevant to the {position} and my skills from the resume.
            2.3 Personalize the introduction, emphasizing your interest in the role.
            2.4 Tailor the body of the letter to reflect your matching skills and experiences.
            2.5 Conclude with a strong, relevant closing statement.
            2.6 Ensure it is under 250 characters. Ensure proper grammar, punctuation, and spacing.
        - Focus: Clarity, relevance, and personalization.
        - Place the output in the key "cover_letter" in the output JSON.

        Step 4: Compose a Professional Email
        - Sub-steps:
            4.1 Based on the job description, draft a professional email to the recruiter or hiring manager with content from the cover letter.
            4.2 Create a placeholder for recruiter's name as Dear [Contact Name]
            4.3 Write a concise email body, mentioning the job link and company name.
            4.4 Develop a subject line that is both relevant and attention-grabbing. It should be under 100 characters.
        - Objective: Clear and professional email communication.
        - Place the output in the keys "email_content" and "email_subject" in the output JSON.

        Step 5: Compose a LinkedIn Note
        -Use the following template:
            Dear [Contact Name],
            I am keen on an open {position} role at {company_name}. I'd appreciate the opportunity to connect and explore how my expertise aligns with this role
        - Place the output in the key "linkedin_note" in the output JSON.

        Output: 
        - Present the output in a JSON format, as per {format_instructions}.
    """


    def __init__(self, llm_args: dict, prompt_args: dict, use_email: bool = True, use_linkedin: bool = True):
        """
        Initializes the connector with LLM and prompt configurations.

        Args:
            llm_args (dict): Configuration for LLM (API key and model name).
            prompt_args (dict): Templates and other arguments for the prompts.
            use_email (bool): Indicates if email template is to be used.
            use_linkedin (bool): Indicates if LinkedIn note template is to be used.
        """
        self._llm_client = ChatOpenAI(
            api_key=llm_args["api_key"],
            model_name=llm_args["model_name"],
        )
        self._prompt_args = prompt_args
        self._use_email = use_email
        self._use_linkedin = use_linkedin

    def generate_custom_content(self, job: pd.Series) -> Dict[str, str]:
        """
        Generates custom content based on the job data.

        Args:
            job (pd.Series): Job data used to generate custom content.

        Returns:
            Dict[str, str]: Generated custom content.
        """
        prompt_args = self._create_prompt_arguments(job)
        output_parser = self._select_output_parser()
        prompt = self._construct_prompt(prompt_args, output_parser)
        chain = prompt | self._llm_client | output_parser

        with get_openai_callback() as callback:
            response_raw = chain.invoke(prompt_args)
            response = response_raw.model_dump()
            logging.info(f"Tokens used: {callback}")

        # Update response with proper keys.
        return {
            "Cover Letter": response["cover_letter"],
            "Resume": response["resume_summary"],
            "Missing Keywords": response["missing_keywords"],
            "Message Content": response["email_content"],
            "Message Subject": response["email_subject"],
            "LinkedIn Note": response["linkedin_note"],
        }

    def _create_prompt_arguments(self, job: pd.Series) -> Dict[str, str]:
        """
        Creates prompt arguments from job data.

        Args:
            job (pd.Series): Job data.

        Returns:
            Dict[str, str]: Arguments for the prompt.
        """
        prompt_args = {
            # "job_description": trim(job["Description"]),
            "job_description": job["Description"],
            "position": job["Position"],
            "company_name": job["Company Name"],
            "name": job['Contact Name'],
            "cover_letter_template": self._prompt_args["cover_letter_template"],
            "resume_template": self._prompt_args["resume_template"],
            "resume_professional_summary": self._prompt_args["resume_professional_summary"],
            # "email_template": self._prompt_args["email_template"] if self._use_email else "",
            # "linkedin_note_template": self._prompt_args["linkedin_note_template"] if self._use_linkedin else "",
        }

        return prompt_args

    @staticmethod
    def _construct_prompt(args: Dict[str, str], output_parser: PydanticOutputParser) -> PromptTemplate:
        """
        Constructs the prompt template.

        Args:
            args (Dict[str, str]): Arguments for the prompt.
            output_parser (PydanticOutputParser): Parser for the LLM response.

        Returns:
            PromptTemplate: Constructed prompt template.
        """
        return PromptTemplate(
            template=LLMConnectorClass.prompt_template,
            input_variables=list(args.keys()),
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )

    @staticmethod
    def _select_output_parser() -> PydanticOutputParser:
        """
        Selects the appropriate output parser.

        Returns:
            PydanticOutputParser: Output parser for the LLM response.
        """
        return PydanticOutputParser(pydantic_object=CustomJobApplicationMaterials)

    @property
    def llm_client(self) -> ChatOpenAI:
        """
        Returns the LLM client instance.

        Returns:
            ChatOpenAI: LLM client instance.
        """
        return self._llm_client
