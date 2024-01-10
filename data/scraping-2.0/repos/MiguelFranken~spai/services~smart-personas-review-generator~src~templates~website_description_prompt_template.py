from textwrap import dedent
from langchain import PromptTemplate


def generate_website_description_prompt_template():
    template = """
    You are provided with a website's metadata details, including its title, description, and URL.
    Analyze and identify the likely name of the website. Determine the overarching purpose of the website and
    provide a high-level explanation of why a user might visit it. Based on the structure of the title and the URL,
    ascertain if the user is on the main landing page or a specific subpage, such as an article or product. If it's a
    subpage, succinctly describe its nature in one sentence and explain why this page might be useful for a user in
    the broader context of the website's purpose. Avoid advocating for the product's qualities; instead, focus on the
    page's utility in the context of the website's mission.

    URL: {website_metadata_url}

    Title: {website_metadata_title}

    Description: {website_metadata_description}

    Other Metadata:
    {website_metadata}

    Answer:
    """

    prompt_template = PromptTemplate(
        input_variables=[
            "website_metadata_url",
            "website_metadata_title",
            "website_metadata_description",
            "website_metadata"
        ],
        template=dedent(template).strip()
    )

    return prompt_template
