from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def compute_constants():
    schema_queries = {
        "additional_translations": (
            "Create additional English translations that are same in meaning but are different from "
            "<Expression>."
        ),
        "explanation_of_expression": (
            "Create 1 sentence long explanation for "
            "<Expression>."
        ),
        # if it is not easily seeable object, try to use explanation_of_expression
        "image_description0": (
            "Create image text prompt that will be used to generate image. "
            "The image has to simply describe <Expression>. "
            "If <Expression> is not visible object, use <explanation_of_expression> to generate a scenario that explains it. "
            "The text must not use any of the following objects: "
            "people; person; persona; professional; hands; fingers; face; poster; signs; letters; text"
        ),
        "image_description1": (
            "Create image description that describes <Expression> according to this explanation <explanation_of_expression>"
            "\n\nMake sure you fulfill these requirements: "
            "\n - Description must simply express the <Expression> with visible objects."
            "\n - Description must not use any of the following objects: "
            "people; person; persona; professional; hands; fingers; face; poster; signs; letters; text"
        ),
        "image_description2": (
            "For this task, I want you to act as a text prompt generator for creating image with AI. "
            "Your main job is to express <explanation_of_expression> with emphasis on <Expression>."
            "\n\nInstruction 1 - provide at least 3 sentences long, detailed image description"
            "\nInstruction 2 - in the description, you must not use any of the following objects:"
            "\npeople; person; persona; professional; human; hands; face; poster; signs; letters; text"
            "\n\nAfter the description is created, check whether the Instruction 2 was fulfilled. "
            "If not, create another image description according to all Instructions."
        ),
        "image_description3": (
            "Create image description that simply explains <Expression> with an image. "
            "\n\nInstruction 1 - if <Expression> is visible object, use only one sentence long description. "
            "Otherwise use at least 3 sentences long and detailed image description."
            "\nInstruction 2 - in the description, you must not use any of the following objects:"
            "\npeople; person; persona; professional; human; hands; face; poster; signs; letters; text"
            "\n\nAfter the description is created, check whether the Instruction 2 was fulfilled. "
            "If not, create another image description according to all Instructions."
        ),
        "image_tags": (
            "Create 10 tags that describe the background details that are closely related with: "
            "<explanation_of_expression>. Use adjactives only!"
        ),
        "explanation_of_syllables": (
            "Explain the meaning of each syllable (only if it has meaning) in "
            "<Expression>. Example - For word 생년월일 we use this format: "
            "생 - birth (it represents the concept of someone coming into existence or being born), 년 - year (it refers to a specific period of time), 월 - month, 일 - day"
        ),
        "sentence_example": (
            "Show how the "
            "<Expression> can be used in a simple korean sentence. "
            "Example - for word 죄송하다 we use this format: "
            "오늘은 친구에게 죄송하다고 사과했어요. (Today, I apologized to my friend.)"
        ),
    }

    response_schemas = []
    queries = ""
    for schema, query in schema_queries.items():
        response_schemas.append(ResponseSchema(name=schema, description=""))
        queries += f"<{schema}>:\n{query}\n\n"

    review_template = f"""
    I want to create additional information of words in my vocabulary list.

    {queries}""" + """
    Do it for this word:
    <Expression>: {korean}, <Translation>: {translation}

    {format_instructions}
    """

    llm = ChatOpenAI(temperature=0.0)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_template(template=review_template)

    return prompt, format_instructions, llm, output_parser

def clean_tags(output_dict):
    image_tags_no_dupl = ""
    img_descr_lower = output_dict["image_description1"].lower()
    for i, tag in enumerate(output_dict["image_tags"].split(",")):
        if tag.lower() not in img_descr_lower:
            image_tags_no_dupl += f"{tag},"
        if i > 4:
            break
    
    output_dict["image_tags"] = image_tags_no_dupl

    return output_dict



def get_addition_word_data(korean, translation):
    messages = PROMPT.format_messages(
        korean=korean,
        translation=translation,
        format_instructions=FORMAT_INSTRUCTIONS,
    )
    response = LLM(messages)
    output_dict = OUTPUT_PARSER.parse(response.content)
    output_dict = clean_tags(output_dict)

    return output_dict

PROMPT, FORMAT_INSTRUCTIONS, LLM, OUTPUT_PARSER = compute_constants()

# # for Tokens used printing
# from langchain.callbacks import get_openai_callback
# with get_openai_callback() as cb:
#     response = LLM(messages)
#     print(cb)


# "image_description0": (
#     "Create image description that describes <Expression> according to this explanation <explanation_of_expression>"
#     "\n\nMake sure you fulfill these requirements: "
#     "\n - Description must catch the essence of <Expression> with objects in a simple way."
#     "\n - Description must be 3 sentences long."
#     "\n - Description must not use any of the following objects: "
#     "people; person; persona; professional; hands; fingers; face; poster; signs; letters; text"
# ),

# "image_description0": (
#             "Create image description that describes <Expression> according to this explanation <explanation_of_expression>"
#             "\n\nMake sure you fulfill these requirements: "
#             "\n - Description must simply express the <Expression> with visible objects."
#             "\n - Description must not use any of the following objects: "
#             "people; person; persona; professional; hands; fingers; face; poster; signs; letters; text"
#         ),