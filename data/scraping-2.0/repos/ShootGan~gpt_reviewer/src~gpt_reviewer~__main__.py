# pylint: disable=missing-module-docstring
import logging
import argparse
import os
import xml.etree.ElementTree as ET
from openai import OpenAI, types


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
    handlers = [ logging.StreamHandler()]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_file(file_path:str) -> str:
    """
    :param file_path: path to file
    :type file_path: str
    :raises FileNotFoundError:
    :return: file content
    :rtype: str
    """

    normalized_path = os.path.normpath(file_path)
    if not os.path.isfile(normalized_path):
        logger.error('File %s not found', normalized_path)
        raise FileNotFoundError(f'File {normalized_path} not found')
    file_content_with_line_numbers =""
    with open(normalized_path, 'r') as file:
        lines = file.readlines()
        file_content_with_line_numbers = ''.join(f"{i + 1}: {line}" for i, line in enumerate(lines))
    logger.info('File %s loaded', normalized_path)
    return file_content_with_line_numbers

def get_review(file_content:str, open_ai_instance:OpenAI ) -> str:
    """
    :param file_content: content of file to review
    :type file_content: str
    :param open_ai_instance: instance of open ai
    :type open_ai_instance: object
    :return: review
    :rtype: object
    """
    xml_template ="""
    <CodeReview>
        <GeneralObservations>
            <Observation>
                General observations about the code's structure, style, programming practices, etc.
            </Observation>
            <!-- Additional general observations can be added here -->
        </GeneralObservations>
        <SpecificComments>
            <Comment>
                <Location> Specify the particular line or section of the code </Location>
                <Feedback>
                    Detailed comment regarding the specific line or section of the code
                </Feedback>
            </Comment>
            <!-- Additional specific comments can be added here -->
        </SpecificComments>
        <ImprovementSuggestions>
            <Suggestion>
                Specific suggestions for improving the code
            </Suggestion>
            <!-- Additional suggestions for improvements can be added here -->
        </ImprovementSuggestions>
    </CodeReview>
    """
    language = "English"
    completion = open_ai_instance.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a program that reviews code and returns feedback in the specified XML format.I ncluding specific examples of issues. Please use the provided XML template for your response. The language of the review should be: {language}. The XML template is: ```{xml_template}```"
            },
            {
                "role": "user",
                "content": f"{file_content}",
            },
        ],
    )
    return completion.choices[0].message.content


def add_content_to_html(html_template:str, root:ET) -> str:
    """
    :param html_template: html template
    :type html_template: str
    :param root: root of xml
    :type root: object
    :return: html with content
    :rtype: str
    """
    # Start building HTML content
    html_report = html_template

    # General Observations
    html_report += "<h2>General Observations</h2><ul>"
    for observation in root.find('GeneralObservations'):
        html_report += f"<li>{observation.text}</li>"
    html_report += "</ul>"

    # Specific Comments
    html_report += "<h2>Specific Comments</h2><ul>"
    for comment in root.find('SpecificComments'):
        location = comment.find('Location').text
        feedback = comment.find('Feedback').text
        html_report += f"<li><strong>{location}:</strong> {feedback}</li>"
    html_report += "</ul>"

    # Improvement Suggestions
    html_report += "<h2>Improvement Suggestions</h2><ul>"
    for suggestion in root.find('ImprovementSuggestions'):
        html_report += f"<li>{suggestion.text}</li>"
    html_report += "</ul>"

    # End HTML report
    html_report += "</div></body></html>"
    return html_report

def generate_report(review:str, output_dir:str):
    """
    :param review: review to save
    :type review: str
    :param output_dir: directory where report will be created
    :type output_dir: str
    """
    root = ET.fromstring(review)
    html_template_path = os.path.join(os.path.dirname(__file__), 'templates', 'code_review_template.html')
    with open(html_template_path, 'r') as file:
        html_template = file.read()
    final_report = add_content_to_html(html_template, root)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'code_review.html')
    with open(output_file, 'w') as file:
        file.write(final_report)
    logger.info('Report saved in %s', output_file)


def main(input_file:str, output_dir:str):
    """
    :param input_file: file to review
    :type input_file: str
    :param output_dir: directory where report will be created
    :type output_dir: str
    """
    logger.info('Creating OpenAI client')
    open_ai_client = OpenAI() #API Token should be in env variable: 'OPENAI_API_KEY'
    logger.info('Getting file content')
    file_content = get_file(input_file)
    logger.info('Getting review')
    review = get_review(file_content, open_ai_client)
    logger.info('Generating report')
    generate_report(review, output_dir)
    logger.info('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog= 'gpt_reviewer',
        description= 'Review code using GPT'
    )
    parser.add_argument(
        "-i",
        "--input_file",
        help= 'Path to file to review',
        type= str,
        required= True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help= 'Path to output directory',
        type= str,
        required= True
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help= 'Increase output verbosity  to debug',
        action= 'store_true'
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Verbose mode activated')

    main(
        input_file= args.input_file,
        output_dir= args.output_path,
    )
