
import fitz
from PIL import Image
import io
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import base64


def extract_image_from_pdf(pdf_path):

    box = (0, 0, 760, 510)
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Get the first page
    page = pdf_document[0]

    # Get image list - this gets a list of XREF of images
    img_list = page.get_images(full=True)

    # For this example, we take the first image
    xref = img_list[0][0]
    base_image = pdf_document.extract_image(xref)
    image_bytes = base_image["image"]

    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Crop the image
    cropped_image = pil_image.crop(box)

    byte_arr = io.BytesIO()
    cropped_image.save(byte_arr, format='PNG')

    # Convert bytes to base64 string
    base64_image = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    return base64_image


def recipeFetcher(query: str, retriever):
    recipe = retriever.get_relevant_documents(query)
    recipe = recipe[0]
    source_pdf_path = recipe.metadata.get('source')

    image = extract_image_from_pdf(source_pdf_path)

    return {'recipe_content': recipe.page_content, 'recipe_path': source_pdf_path, 'recipe_image': image}


def recipeJson(recipe, llm):

    json_template = """You will generate valid json using the following format based on recipe data you are given. Only generate the json_data as I am using this output to parse another function.

    they keys for json_data are: title, author, cooktime

    the author should be the first author in the recipe data, and not include the word "by"

    recipe data = {recipe_data}

    recipe json ="""

    prompt_template = PromptTemplate(
        input_variables=["recipe_data"], template=json_template)
    recipe_json_chain = LLMChain(llm=llm, prompt=prompt_template)

    recipe_json_str = recipe_json_chain.run(recipe_data=recipe)

    print(recipe_json_str)

    try:
        recipe_json = json.loads(recipe_json_str)
    except:
        recipe_json = 'Error in generating json'

    return recipe_json


def recipeSummarizer(query, recipe_data, llm):
    template_recipe_reader = """ You will be given the data of a single recipe. You will read the recipe and use only the information there to answer the user.
You will also be given a user prompt to answer. You will answer the user prompt based on the recipe data.

recipe data: {recipe_data}
user prompt: {user_prompt}

Recipe Response to user:
"""

    prompt_template_recipe_reader_v2 = PromptTemplate(
        input_variables=["recipe_data", "user_prompt"], template=template_recipe_reader)
    recipe_reader_chain_v2 = LLMChain(
        llm=llm, prompt=prompt_template_recipe_reader_v2)

    recipe_reader_response = recipe_reader_chain_v2.run(
        recipe_data=recipe_data, user_prompt=query)

    return recipe_reader_response


def recipeOrchestrator(query: str, retriever, llm):
    recipe = recipeFetcher(query, retriever)
    recipe_json = recipeJson(recipe['recipe_content'], llm)
    recipe['recipe_json'] = recipe_json
    recipe_summary = recipeSummarizer(query, recipe['recipe_content'], llm)
    recipe['recipe_summary'] = recipe_summary

    return recipe
