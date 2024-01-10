from flask import Blueprint, make_response, jsonify
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import asyncio

load_dotenv()
client = AsyncOpenAI()
categorize = Blueprint("categorize", __name__)


@categorize.route("/healthcheck")
def healthcheck():
    return make_response("OK", 200)


@categorize.route("/categorize", methods=["POST"])
def categorize_endpoint():
    """Given a transcript, return the question type and Costa's level of reasoning for each question.

    Returns:
        dict: A dictionary containing the question type and Costa's level of reasoning for each question.
    """

    pass


def validate_category_output(output):
    """
    Validate the output of the categorize function.

    Args:
        output (dict): Output of the categorize function.

    Returns:
        (bool, str): A tuple containing a boolean indicating whether the output is valid and an error message if is not valid.
    """

    # Ensure the output JSON has the required fields
    required_fields = ["question_type", "question_level"]
    if not all(field in output for field in required_fields):
        return (False, "Output JSON must contain question_type and question_level")
    # Ensure the question level is an integer
    if not isinstance(output["question_level"], int):
        return (False, "question_level must be an integer")
    # Ensure the question level is between 0 and 3
    if not 0 <= output["question_level"] <= 3:
        return (False, "question_level must be between 0 and 3")
    # Ensure the question type is a string and is one of the allowed values
    if not isinstance(output["question_type"], str):
        return (False, "question_type must be a string")
    allowed_question_types = [
        "Knowledge",
        "Analyze",
        "Apply",
        "Create",
        "Evaluate",
        "Understand",
        "Rhetorical",
        "Unknown",
    ]
    if output["question_type"] not in allowed_question_types:
        return (False, f"question_type must be one of {allowed_question_types}")

    return True, None


async def categorize_question(data) -> dict:
    """
    Categorize the question type and Costa's level of reasoning of a question given the context, using GPT-4.

    Args:
        summary (str): Summary of the question.
        previous_sentence (str): Previous sentence in the context.
        previous_speaker (str): Speaker of the previous sentence.
        question (str): Question to categorize.
        question_speaker (str): Speaker of the question.
        next_sentence (str): Next sentence in the context.
        next_speaker (str): Speaker of the next sentence.

    Returns:
        dict: A dictionary containing the question type and Costa's level of reasoning.
    """

    # Basic input validation
    required_fields = [
        "summary",
        "previous_sentence",
        "previous_speaker",
        "question",
        "question_speaker",
        "next_sentence",
        "next_speaker",
    ]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Construct the messages to send to the language model

    system_role = """
    Given the following context and question from the speaker, determine the question type and Costa's level of reasoning.
    The question type should be categorized as Knowledge, Analyze, Apply, Create, Evaluate, Understand, Rhetorical, or Unknown.
    Costa's levels of reasoning should be categorized as 1 (gathering), 2 (processing), 3 (applying), or 0 (n/a).
    Provide the analysis in JSON format as specified.
    --- BEGIN USER MESSAGE ---
        Context: "$SUMMARY"
        Previous Sentence: "$PREVIOUS_SENTENCE"
        Speaker of Previous Sentence: "$PREVIOUS_SPEAKER"
        Question: "$QUESTION"
        Speaker of Question: "$QUESTION_SPEAKER"
        Next Sentence: "$NEXT_SENTENCE"
        Speaker of Next Sentence: "$NEXT_SPEAKER"
    --- END USER MESSAGE ---
    Analyze the question and output the results in the following JSON format,
    where QUESTION_TYPE is a str and QUESTION_LEVEL is an int (1, 2, 3, or 0):

    ---BEGIN FORMAT---
    {
        "question_type":"$QUESTION_TYPE",
        "question_level":"$QUESTION_LEVEL"
    }
    ---END FORMAT---
    """

    user_message = f"""
    Context: {data['summary']}
    Previous Sentence: {data['previous_sentence']}
    Speaker of Previous Sentence: {data['previous_speaker']}
    Question: {data['question']}
    Speaker of Question: {data['question_speaker']}
    Next Sentence: {data['next_sentence']}
    Speaker of Next Sentence: {data['next_speaker']}
    """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_message},
    ]

    # Call the OpenAI API with the prompt
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
    )

    print(str(response.choices[0].message.content))

    # Extract the response and format it as JSON
    output = str(response.choices[0].message.content)

    # Parse the output JSON
    output = json.loads(output)

    # Validate the output JSON
    valid, error = validate_category_output(output)
    if not valid:  # If the output JSON is not valid, return that we don't know the question type or level
        return {
            "question_type": "Unknown",
            "question_level": 0,
        }

    # Return the output JSON
    return output


# async def test_categorize_question():
#     """
#     Test the categorize_question function.
#     """

#     # Define the sample input data
#     sample_data = {
#         "summary": "Context summary",
#         "previous_sentence": "Previous sentence",
#         "previous_speaker": "Speaker A",
#         "question": "What is the capital of France?",
#         "question_speaker": "Speaker B",
#         "next_sentence": "Next sentence",
#         "next_speaker": "Speaker C",
#     }

#     tasks = [categorize_question(sample_data) for _ in range(10)]
#     results = asyncio.run(asyncio.gather(*tasks))

#     print(results)

# asyncio.run(test_categorize_question())

sample_data = {
    "summary": "Context summary",
    "previous_sentence": "Previous sentence",
    "previous_speaker": "Speaker A",
    "question": "What is the capital of France?",
    "question_speaker": "Speaker B",
    "next_sentence": "Next sentence",
    "next_speaker": "Speaker C",
}

loop = asyncio.get_event_loop()
ret_val = loop.run_until_complete(categorize_question(sample_data))
print(ret_val)
