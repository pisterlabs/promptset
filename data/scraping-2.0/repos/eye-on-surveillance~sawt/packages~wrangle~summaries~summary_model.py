import pytesseract
from pdf2image import convert_from_path
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
import json
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import uuid
import re


def pdf_to_images(pdf_path):
    """Converts PDF file to images"""
    return convert_from_path(pdf_path)


def extract_text_from_image(image):
    """Extracts text from a single image using OCR"""
    return pytesseract.image_to_string(image)


def save_ocr_to_json(pdf_path, ocr_json_path, publish_date):
    """Performs OCR on a PDF and saves the result in a JSON format"""
    images = pdf_to_images(pdf_path)
    messages = [{"page_content": extract_text_from_image(image)} for image in images]

    with open(ocr_json_path, "w") as file:
        json.dump({"messages": messages}, file, indent=4)


def load_and_split(json_path, chunk_size=4000, chunk_overlap=1000):
    """Loads OCR text from JSON and splits it into chunks that approximately span 2 pages"""
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(data)


def extract_date_from_filename(filename):
    """Extracts the publish date from the PDF filename using regex"""
    match = re.search(r"\d{1,2}-\d{1,2}-\d{4}", filename)
    return match.group(0) if match else None


def summarize_text(chunks, publish_date):
    """Summarizes the chunks of text"""
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        api_key="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",
    )
    summaries = []

    for chunk in chunks:
        text_content = chunk.page_content
        uid = str(uuid.uuid4())

        prompt = PromptTemplate(
            input_variables=["text_content", "uid"],
            template="""
        ## Council Meeting Ordinance Summary

        ### Ordinance Details and Voting Outcomes:
        {text_content}

        ### Summary Guidelines:
        - **Objective**: Clearly summarize each ordinance that was up for vote, including its brief description and the outcome of the vote (whether it passed or not).
        - **Structure**: Present each ordinance separately, starting with its calendar number and title, followed by a brief description, the voting results, and any noteworthy amendments or discussions.
        - **Detail**: Highlight important aspects of each ordinance, such as the purpose of the ordinance, key amendments, and the final decision (passed, amended, withdrawn, etc.).
        - **Formatting**: Use a structured format, listing each ordinance as a separate bullet point for clarity.
        - **Tone**: Maintain a neutral and factual tone, focusing on delivering information as presented in the chunk.

        ### Additional Instructions:
        - **Specificity**: Ensure the summary is specific to the content of each ordinance, avoiding general statements.
        - **Contextual Clarity**: Where necessary, provide context to clarify the purpose of the ordinance or the implications of the vote.
        - **Coherence**: Each summary should provide a complete understanding of the ordinance's discussion and outcome within the council meeting.
        - For each ordinance, summarize the content, identify the ordinance number, which council member introduced it, identify the topic, and include the generated UID: {uid}.

        ### Example Format:
        - Topic: [Primary topic or focus of this chunk]]
        - Summary: [Your summary here]
        - Ordinance Number: [Ordinance number here]
        - Votes Summary:
            Vote 1: Passed or Failed or N/A - (Number of YEAS, Number of NAYS, Number of ABSTAIN, Number of ABSENT)
            Vote 2: [Summary of the second vote, if applicable]
            ...(Continue for additional votes)
        - Decision/Key Actions: [Key decisions or actions]
        - Tags/Keywords: [Relevant tags or keywords]
        - UID: {uid}

        ### Role Emphasis:
        As an AI assistant, your task is to distill key information from the meeting's minutes, offering clear and concise summaries of each ordinance and motion, and their respective outcomes, to enable quick understanding and retrieval of crucial details.
        """,
        )

        chain = LLMChain(llm=chat, prompt=prompt)
        summary = chain.run(text_content=text_content, uid=uid, temperature=1)
        print(summary)

        summaries.append(
            {"page_content": summary, "uid": uid, "publish_date": publish_date}
        )
    return summaries


def save_summaries_to_json(summaries, output_dir, pdf_filename):
    """Saves the summaries to a JSON file, with all summaries under the key 'messages'"""
    output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_filename)[0]}.json")
    with open(output_file, "w") as file:
        json.dump({"messages": summaries}, file, indent=4)


def concatenate_jsons(input_dir, output_file):
    all_messages = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)

            with open(file_path, "r") as file:
                data = json.load(file)
                messages = data.get("messages", [])

                all_messages.extend(messages)

    with open(output_file, "w") as file:
        json.dump({"messages": all_messages}, file, indent=4)


# if __name__ == "__main__":
#     documents_directory = "../input"
#     output_json_dir = "../output"

#     os.makedirs(output_json_dir, exist_ok=True)  #

#     for pdf_filename in os.listdir(documents_directory):
#         if pdf_filename.endswith(".pdf"):
#             output_json_path = os.path.join(
#                 output_json_dir, f"{os.path.splitext(pdf_filename)[0]}.json"
#             )

#             if os.path.exists(output_json_path):
#                 print(f"Skipping {pdf_filename}, output already exists.")
#                 continue

#             pdf_path = os.path.join(documents_directory, pdf_filename)
#             publish_date = extract_date_from_filename(pdf_filename)
#             ocr_json_path = "../output/ocr_text.json"

#             save_ocr_to_json(pdf_path, ocr_json_path, publish_date)
#             chunks = load_and_split(ocr_json_path)
#             summaries = summarize_text(chunks, publish_date)

#             save_summaries_to_json(summaries, output_json_dir, pdf_filename)
#             os.remove(ocr_json_path)

#     input_directory = "../output"
#     output_json_path = "../output/Minutes 2022.json"
#     concatenate_jsons(input_directory, output_json_path)
#     print(f"Summaries saved in directory: {output_json_dir}")
