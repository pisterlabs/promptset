import numpy as np

import cv2
import pytesseract

import easyocr


from pprint import pprint

import os
from dotenv import load_dotenv
import openai

load_dotenv()




pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\laure\\Tesseract-Temp\\tesseract.exe"




class Text:

    def __init__(self, top, left, height, width, value):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.bottom = self.top + self.height
        self.right = self.left + self.width
        self.value = value

class Button:

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.bottom = self.top + self.height
        self.right = self.left + self.width
        self.text = None

    def has_text(self, Text):
        # print(self.top)
        # print(Text.top)
        if self.top < Text.top:
            if self.left < Text.left:
                if self.right > Text.right:
                    if self.bottom > Text.bottom:
                        self.text = Text.value
                        return True
        return False
    
    def return_json(self):
        return {
                    "width": self.width,
                    "height": self.height,
                    "value": self.text,
                    "marginTop": self.top,
                    "marginLeft": self.left,
                    "fontSize": 12
            }
    
    def __repr__(self):
        return f"height: {self.height}; width: {self.width}; top: {self.top}; left: {self.left}; text: {self.text}"












reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
results = reader.readtext('static/img_buttons.png')

pprint(results)

list_of_text_objs = []

for result in results:

    width = result[0][1][0] - result[0][0][0]
    # print(width)

    height = result[0][2][1] - result[0][1][1]
    # print(height)

    top = result[0][0][1]
    left = result[0][0][0]

    value = result[1]

    text_obj = Text(top=top, left=left, height=height, width=width, value=value)

    list_of_text_objs.append(text_obj)




image = cv2.imread("static/img_buttons.png")
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



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
    # pprint("This is a contour of the bounding rectangle")
    # pprint(bounding_rectangle)
    bound_obj = Button(top=bounding_rectangle[1], left=bounding_rectangle[0], width=bounding_rectangle[2], height=bounding_rectangle[3])

    for text_obj in list_of_text_objs:
        if bound_obj.has_text(text_obj):
            pprint(bound_obj)
            pprint(bound_obj.return_json())

main_list_text = []

text = pytesseract.image_to_string(image)

pprint(">>> text:::")

pprint(text)


