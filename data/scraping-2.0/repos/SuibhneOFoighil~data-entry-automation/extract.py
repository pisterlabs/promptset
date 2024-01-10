import os
import openai
import pypdfium2 as pdfium
from PIL import Image
from io import BytesIO
import pytesseract

# 1. Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

# 2. Extract text from images via pytesseract
def extract_text_from_img(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        try:
            raw_text = str(pytesseract.image_to_string(image))
            image_content.append(raw_text)
        except pytesseract.pytesseract.TesseractError:
            print(f"Error: pytesseract could not read image {index}")
        except Exception as e:
            print(f"Error: {e}")


    return "\n".join(image_content)

def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract


# 3. Extract structured info from text via LLM
def extract_structured_data(content: str, data_points) -> str:

    prompt = f"""
    You are an expert admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format. Your job is to extract the following data points:
    {data_points}

    Now please extract details from the content and export in a JSON array format, 
    return ONLY the JSON array:
    """

    # create openAI endpoint
    result = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return result.choices[0].text