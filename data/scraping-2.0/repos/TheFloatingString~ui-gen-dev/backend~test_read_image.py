import cv2
import pytesseract
from pprint import pprint

import os
from dotenv import load_dotenv
import openai

load_dotenv()

IMAGE_FILEPATH = "static\\IMG_2881.jpg"

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\laure\\Tesseract-Temp\\tesseract.exe"

img = cv2.imread(IMAGE_FILEPATH)


openai.api_key = os.getenv("X_OPENAI_API_KEY")


main_html_elements = []

# Sanity check
pprint(img)


# Preprocessing the image starts
 
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.png", gray)
 
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
 
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

cv2.imwrite("dilation.png", dilation)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
main_list_bounding_rectangles = []

for contour in contours:
    bounding_rectangle = cv2.boundingRect(contour)
    print(bounding_rectangle)

main_list_text = []

text = pytesseract.image_to_string(img)

for individual_line in text.split("\n"):
    if len(individual_line)>0:
        print(individual_line)
        main_list_text.append(individual_line)


print("main_list_text:")
print(main_list_text)


def generate_p_tag(text_descriptor):
    return f"<p>{text_descriptor}</p>"

def get_instructor(text_descriptor):
    potential_instructor = text_descriptor.split(" ")[0]
    if potential_instructor[0] == '#':
        return potential_instructor.strip().replace(":", "")
    else:
        return None

def generate_img_tag(text_descriptor):

    print(f"text descriptor: {text_descriptor}")

    response = openai.Image.create(
        prompt=text_descriptor,
        n=1,
        size="256x256"
        )
    
    image_url = response['data'][0]['url']
    print(image_url)

    return f"<img src='{image_url}'>"


for sample_text in main_list_text:

    print(f"sample_text: {sample_text}")

    instructor = get_instructor(sample_text)

    if instructor.upper() in ["#IMAGE", "#IMG", "#LMG"]:
        main_html_elements.append(generate_img_tag(sample_text))
    else:
        main_html_elements.append(generate_p_tag(sample_text))

    print("printing `main_html_elements`")
    print(main_html_elements)



def generate_html(list_of_html_elements):
    generated_HTML_code = """
        <html>
            <body>
    """

    for element in list_of_html_elements:
        generated_HTML_code += element

    generated_HTML_code += "</body></html>"

    return generated_HTML_code

with open("sample_output.html", "w") as output_file:
    output_file.write(generate_html(main_html_elements))
    output_file.close()