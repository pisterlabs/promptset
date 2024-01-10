from textwrap import dedent
from langchain import PromptTemplate


def generate_website_a11y_report_prompt_template():
    """
    Generates a prompt template to summarize website accessibility violations in natural language.

    This function structures a prompt that takes in an accessibility report (in array format) and provides
    instructions to summarize the report. The resultant summary is intended to highlight the significance of each
    violation for people with disabilities, ensuring that the explanation avoids technical terms as much as possible.

    Returns:
        PromptTemplate: An instance of the PromptTemplate with the specified input variables and structured template.
    """

    template = """
    Given the accessibility report below in array syntax, summarize the accessibility violations of the
    website in natural language. The summary should contain as few technical terms as possible and explain why each
    violation matters for people with disabilities.

    {a11y_extended}

    Answer:
    """

    prompt_template = PromptTemplate(
        input_variables=[
            "a11y_extended",
        ],
        template=dedent(template).strip()
    )

    return prompt_template
