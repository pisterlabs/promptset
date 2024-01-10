import base64
from io import BytesIO
from PIL import Image


def convert_to_base64(image_file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    pil_image = Image.open(image_file_path)

    buffered = BytesIO()
    pil_image.save(buffered, format="png")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    return image_html
    # To display the image by rendering the HTML
    # from IPython.display import HTML, display
    # display(HTML(image_html))




if __name__ == "__main__":

    # https://python.langchain.com/docs/integrations/llms/ollama#multi-modal

    # example
    file_path = "/Users/aziz/Desktop/style.png"
    # pil_image = Image.open(file_path)
    # image_b64 = convert_to_base64(pil_image)
    # plt_img_base64(image_b64)

    image_b64 = convert_to_base64(file_path)
    plt_img_base64(image_b64)


    # create mmodel
    from langchain.llms import Ollama

    bakllava = Ollama(model="bakllava")
  
    # run model
    llm_with_image_context = bakllava.bind(images=[image_b64])
    res = llm_with_image_context.invoke("Describe the image:")

    print(res)