from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=(
            "As an AI document generator, your task is to generate an HTML and CSS document "
            "based on the provided document description. The HTML and CSS should adhere to a clean "
            "and professional design aesthetic. Use vertical space effectively "
            "and evenly distribute content over the entire height of the document."
            "Here are the specific style guidelines to follow:\n\n"
            "- The document should be designed for an A4 format, with an aspect ratio corresponding to the square root of 2 (1:1.4142). "
            "This should be reflected in the layout of the HTML elements and the CSS styles.\n"
            "- The primary font should be Verdana, with 'Times New Roman' used for title elements.\n"
            "- The primary font size should be 16px, with title elements at 70px and subtitles at 32px.\n"
            "- The primary color for text should be black (#000000), with lighter shades used for non-primary text.\n"
            "- The background color should be #f5f5ef.\n"
            "- The document should have a margin to ensure content doesn't touch the edges of the A4 page.\n"
            "- Tables should have their rows evenly spaced, with borders between rows, and headers should be bold.\n\n"
            "Now, based on these style guidelines and the document description provided, generate "
            "the HTML and CSS documents.\n\n"
            "Document Description:\n\n"
            "{description}\n"
            "Please provide the HTML and CSS in a clean and easily readable format. "
            "The content in your output should precisely match the provided document description, "
            "and not contain placeholders or template items.\n\n"
            "Note: Although the document does not need to be fully responsive across various device sizes, "
            "it should still retain a clean, professional appearance when converted to an A4-sized PDF.\n\n"
            "Output Format: The output should contain two parts: 'HTML' and 'CSS'. "
            "The HTML could refer to the CSS as a second file called styles.css. "
            "Each part should start with a title line: 'HTML:' or 'CSS:', followed by the respective code. "
            "Example: \n\nHTML:\n<html>...</html>\n\nCSS:\nbody ... \n\n"
            "Begin!"
        ),
        input_variables=["description"],
    )
)


import os


def document_tool_builder(chat: BaseLanguageModel):
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    document_chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    def document_tool_runner(description):
        llm_output = document_chain.run(description)

        # Split the output by 'CSS:' to separate HTML and CSS parts
        split_output = llm_output.split("CSS:")
        if len(split_output) != 2:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        # Remove 'HTML:' from the first part and strip extra whitespace
        html_output = split_output[0].replace("HTML:", "").strip()
        css_output = split_output[1].strip()

        # If both HTML and CSS are present, write them into respective files
        if html_output and css_output:
            # Create the 'generated' directory if it does not exist
            os.makedirs("generated", exist_ok=True)

            # Define the paths to store the files inside 'generated' folder
            html_path = "generated/index.html"
            css_path = "generated/styles.css"
            with open(html_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_output)
            with open(css_path, "w", encoding="utf-8") as css_file:
                css_file.write(css_output)
            return f"Document generated with success, html_path: '{html_path}', css_path: '{css_path}'"

        # If output is not in the expected format, raise an error
        raise ValueError(f"Could not parse LLM output: `{llm_output}`")

    return document_tool_runner
