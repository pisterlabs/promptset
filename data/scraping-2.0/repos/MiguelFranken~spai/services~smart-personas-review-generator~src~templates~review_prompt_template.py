from textwrap import dedent

from langchain import PromptTemplate


def generate_review_prompt_template():
    """
    Generates a prompt template for a review of a website's accessibility issues.

    Returns:
        PromptTemplate: A prompt template object with input variables for website description, accessibility assessment,
        paragraph limit, persona story, and persona name.
    """

    template = """
    {persona_story}

    Pretend you are {persona_name} reviewing a website.
    It is crucial that you write a review, limited to {paragraph_limit} paragraphs, utilizing first-person language.

    {persona_name} is writing a review about the following website:
    {website_description_summary}

    {persona_name} found the following accessibility issues on the website:
    {website_a11y_report}

    The review elegantly weaves together all information to clearly explain the issues with the website's
    accessibility. Make the explanation comprehensible **to someone who lacks technical expertise**. By creatively
    incorporating all details, you'll effectively convey the accessibility issues.

    The review might be sarcastic and witty, producing creative and funny responses. The goal is to create empathy
    towards accessibility challenges. Think aloud and talk about your emotions. Use {persona_name_possessive} natural
    language and avoid any technical terms. Include anecdotes from {persona_name_possessive} personal and
    professional life into your review. Also, include the type of the website into your review – imagine how
    {persona_name} would use that type of website in her everyday life. You might speak metaphorically.

    Review:
    """

    prompt_template = PromptTemplate(
        input_variables=[
            "website_description_summary",
            "website_a11y_report",
            "paragraph_limit",
            "persona_story",
            "persona_name",
            "persona_name_possessive"
        ],
        template=dedent(template).strip()
    )

    return prompt_template
