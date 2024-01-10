import json
import os
import re
import tempfile

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_fixed
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean, group_broken_paragraphs
from unstructured.documents.elements import (
    CompositeElement,
    Footer,
    Header,
    Image,
    NarrativeText,
    Title,
    Table,
)
from unstructured.partition.auto import partition


load_dotenv()
openai_client = OpenAI()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_completion(**kwargs):
    return openai_client.chat.completions.create(**kwargs)




directory = "pdfs"

for pdf_name in os.listdir(directory):
    pdf_name = os.path.join(directory, pdf_name)
    
    elements = partition(
        filename=pdf_name,
        pdf_extract_images=False,
        pdf_image_output_dir_path=tempfile.gettempdir(),
        skip_infer_table_types=["jpg", "png", "xls", "xlsx"],
        # strategy="hi_res",
    )

    # filtered_elements = [
    #     element
    #     for element in elements
    #     if not (
    #         isinstance(element, Header)
    #         or isinstance(element, Footer)
    #         or isinstance(element, NarrativeText)
    #     )
    # ]

    processname_element = [
        element.text
        for element in elements
        if "ecoinvent 3.8 Dataset Documentation" in element.text
    ]
    match = re.search(r"'(.*?)'", str(processname_element))
    if match:
        process_name = match.group(1)

    # Initialize two flag variables
    found_source_header = False
    found_restriction_text = False

    filtered_elements = []
    for element in elements:
        # If the element is a Header and its text is "source", set the first flag to True
        if isinstance(element, Title) and element.text == "Source":
            found_source_header = True
        # If the element is a NarrativeText and its text is the restriction text, set the second flag to True
        elif (
            isinstance(element, NarrativeText)
            and element.text
            == "Restriction of Use The restrictions of use stipulated in the EULA remain applicable for this pdf documentation. Copyright ecoinvent Association, 2023"
        ):
            found_restriction_text = True
        # If the first flag is True and the second flag is False, add the element to the filtered_elements list
        if found_source_header and not found_restriction_text:
            filtered_elements.append(element)

    for element in filtered_elements:
        if element.text != "":
            element.text = group_broken_paragraphs(element.text)
            element.text = clean(
                element.text,
                bullets=False,
                extra_whitespace=True,
                dashes=False,
                trailing_punctuation=False,
            )
        # elif isinstance(element, Image):
        #     point1 = element.metadata.coordinates.points[0]
        #     point2 = element.metadata.coordinates.points[2]
        #     width = abs(point2[0] - point1[0])
        #     height = abs(point2[1] - point1[1])
        #     if width >= min_image_width and height >= min_image_height:
        #         element.text = vision_completion(element.metadata.image_path)

    chunks = chunk_by_title(
        elements=filtered_elements,
        multipage_sections=True,
        combine_text_under_n_chars=0,
        new_after_n_chars=None,
        max_characters=4096,
    )

    text_list = []
    for chunk in chunks:
        if isinstance(chunk, CompositeElement):
            text = chunk.text
            text_list.append(text)
        elif isinstance(chunk, Table):
            if text_list:
                text_list[-1] = text_list[-1] + "\n\n" + chunk.metadata.text_as_html
            else:
                text_list.append(chunk.metadata.text_as_html)
    
    with open("test_list.txt", "a+") as f:
        f.write("\n"+str(text_list))

    # result_list = []

    # for text in text_list:
    #     split_text = text.split("\n\n", 1)
    #     if len(split_text) == 2:
    #         title, body = split_text
    #         result_list.append({"title": title, "body": body})

    # msgs = [
    #     SystemMessage(
    #         content="Generate JSON based on the text below."
    #     ),
    #     HumanMessage(content="Text:"),
    #     HumanMessagePromptTemplate.from_template("{result_list}"),
    # ]

    response = create_completion(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON with the key name of CSV_Content.",
            },
            {
                "role": "user",
                "content": f"""从下面信息中仔细分辨并提取信息: First Author, Additional Author(s), Title, Year, Volume Number, Issue Number, Journal，输出为csv格式：\n\n{str(text_list)}""",
            },
        ],
    )

    result = response.choices[0].message.content


    dict_data = json.loads(result)
    # Split the CSV content into lines
    if "CSV_Content" in dict_data:
        lines = dict_data["CSV_Content"].split("\n")

        # Process each line
        for i in range(len(lines)):
            # Skip the header line
            if i == 0:
                continue
            # Add processname and filename to the beginning of the line
            lines[i] = f"{process_name},{pdf_name}," + lines[i]

        lines = lines[1:]
        # Join the lines back into a single string
        new_csv_content = "\n".join(lines)

    # Write the new CSV content to the file
    with open("test.csv", "a+") as f:
        f.write("\n"+new_csv_content)


