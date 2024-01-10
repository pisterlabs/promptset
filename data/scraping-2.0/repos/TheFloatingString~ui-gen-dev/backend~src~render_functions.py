import openai
import easyocr

import string
import random

from dotenv import load_dotenv
import os

from pprint import pprint
import math

import cv2

load_dotenv()


openai.api_key = os.getenv("X_OPENAI_API_KEY")



class Contour:

    def __init__(self, contour):
        self.contour = contour
        self.left = min(contour[:,0,0])
        self.width = max(contour[:,0,0]) - min(contour[:,0,0])
        self.top = min(contour[:,0,1])
        self.height = max(contour[:,0,1]) - min(contour[:,0,1])

    def return_element(self):
        if self.width < 80:
            return RadioButton(top=self.top, left=self.left, height=self.height, width=self.width, value=None)
    
        else:
            return Button(top=self.top, left=self.left, height=self.height, width=self.width)




class Text:

    def __init__(self, top, left, height, width, value):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.bottom = self.top + self.height
        self.right = self.left + self.width
        self.value = value
        self.is_nullified = False

    def nullify(self):
        self.is_nullified = True

    def return_json(self):
        return {
            "type":"Typography",
            "width": self.width,
            "height": self.height,    
            "left_margin": self.left,
            "top_margin": self.top,
            "font_size":"12",
            "text": self.value
        }

    def __repr__(self):
        return f"top: {self.top}; left: {self.left}; text: {self.value}"

class Button:

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.bottom = self.top + self.height
        self.right = self.left + self.width
        self.text = None
        self.element_type="Button"

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
                    "type": "Button",
                    "fontSize": 12
            }
    
    def __repr__(self):
        return f"height: {self.height}; width: {self.width}; top: {self.top}; left: {self.left}; text: {self.text}"


class RadioGroup:

    def __init__(self):
        self.top = None
        self.left = None
        self.height = None
        self.width = None
        self.bottom = None
        self.right = None
        self.initial_dims_updated = False
        self.list_of_radio_button_obj = []
        self.title = "Form"
        self.element_type = "RadioGroup"

    def __repr__(self):
        print("Start of content")
        for button_obj in self.list_of_radio_button_obj:
            pprint(button_obj)
        return "End of content"

    def add_radio_button_to_list(self, radio_button_obj):
        self.list_of_radio_button_obj.append(radio_button_obj)
        # print("call!")
        # print(self.list_of_radio_button_obj)
        if self.initial_dims_updated:
            pass

        else:
            self.top = radio_button_obj.top
            self.height = radio_button_obj.height
            self.left = radio_button_obj.left
            self.width = radio_button_obj.width
        
        self.update_bottom()
        self.update_right()

    def update_bottom(self):
        self.bottom = self.top + self.height

    def update_right(self):
        self.right = self.left + self.width

    def add_button(self, radio_button_obj):
        if len(self.list_of_radio_button_obj) == 0:
            self.add_radio_button_to_list(radio_button_obj=radio_button_obj)

        else:
            if abs(self.left - radio_button_obj.left) < 10:
                if abs(self.top - radio_button_obj.top) < 250:
                    self.add_radio_button_to_list(radio_button_obj=radio_button_obj)

    def return_json(self):
        return_dict = {
            "type":"RadioGroup",
            "title": self.title,
            "options": [radio_button_obj.value for radio_button_obj in self.list_of_radio_button_obj],
            "width": self.width,
            "height": self.height,
            "left_margin": self.left,
            "top_margin": self.top
        }
        
        return return_dict

class RadioButton:

    
    def __init__(self, top, left, height, width, value):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.bottom = self.top + self.height
        self.right = self.left + self.width
        self.value = value
        self.center = (self.left+0.5*self.width, self.top+0.5*self.height)


    def __repr__(self) -> str:
        return f"Center of radio button: {self.center}; text: {self.value}"

    def is_same_button(self, other_button):
        euclidean_dist = math.sqrt((self.center[0] - other_button.center[0])**2 + (self.center[1] - other_button.center[1])**2)
        pprint(f"Euclidean dist: {euclidean_dist}")
        if euclidean_dist <5:
            return True
        else:
            return False
        
    def text_belongs(self, text_obj):

        # pprint(text_obj.top < self.center[1] < text_obj.bottom)
        # pprint(0<(text_obj.left - self.right)<20)
        # print()

        if (text_obj.top < self.center[1] < text_obj.bottom) and (0<(text_obj.left - self.right)<20):
            self.value = text_obj.value
            return True
        else:
            return False





def generate_p_tag(text_descriptor):
    return f"<p>{text_descriptor}</p>"

def get_instructor(text_descriptor):
    potential_instructor = text_descriptor.split(" ")[0]
    print("potential_instructor")
    print(potential_instructor)
    if potential_instructor[0] == '#':
        return potential_instructor.strip().replace(":", "")
    else:
        return None
    
def get_openai_link(text_descriptor):

    response = openai.Image.create(
        prompt=text_descriptor,
        n=1,
        size="256x256"
        )
    
    return response["data"][0]["url"]

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






def screenshot_to_json(image_filepath):

    main_return_json = {"data": {}}


    list_of_text_objs = []

    image = cv2.imread(image_filepath)
    mod_image = cv2.imread(image_filepath)


    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    results = reader.readtext(image_filepath)


    for result in results:

        width = result[0][1][0] - result[0][0][0]

        height = result[0][2][1] - result[0][1][1]

        top = result[0][0][1]
        left = result[0][0][0]

        value = result[1]

        text_obj = Text(top=top, left=left, height=height, width=width, value=value)
        mod_image = cv2.rectangle(mod_image, (text_obj.left,text_obj.top), (text_obj.right,text_obj.bottom), (255,255,255), -1)

        list_of_text_objs.append(text_obj)

    for text_obj in list_of_text_objs:
        pprint(text_obj)



    gray = cv2.cvtColor(mod_image, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
    
    print(f"Length of contours: {len(contours)}")

    list_of_radio_buttons = []
    list_of_buttons = []


    for contour in contours:

        contour_obj = Contour(contour)
        element_obj = contour_obj.return_element()

        if isinstance(element_obj, Button):
            list_of_buttons.append(element_obj)
        
        elif isinstance(element_obj, RadioButton):
            list_of_radio_buttons.append(element_obj)


    final_radio_obj_list = []

    for radio_obj in list_of_radio_buttons:
        for text_obj in list_of_text_objs:
            if radio_obj.text_belongs(text_obj):
                final_radio_obj_list.append(radio_obj)
                text_obj.nullify()
                break

    # for radio_obj in final_radio_obj_list:
    #     pprint(radio_obj)

    radio_group_obj = RadioGroup()

    for button_obj in final_radio_obj_list:
        radio_group_obj.add_button(button_obj)

    if len(radio_group_obj.list_of_radio_button_obj) > 0:
        res = ''.join(random.choices(string.ascii_uppercase +
                            string.digits, k=10))
        main_return_json["data"][res] = radio_group_obj.return_json()

    

    for button_obj in list_of_buttons:
        print(button_obj)
        for text_obj in list_of_text_objs:
            if text_obj.is_nullified == False:
                button_obj.has_text(text_obj)
                if button_obj.text is not None:
                    text_obj.nullify()


    for button_obj in list_of_buttons:
        res = ''.join(random.choices(string.ascii_uppercase +
                            string.digits, k=10))
        main_return_json["data"][res] = button_obj.return_json()

   

   


    # for button_obj in list_of_radio_buttons:
    #     pprint(button_obj)
    #     res = ''.join(random.choices(string.ascii_uppercase +
    #                         string.digits, k=10))
    #     main_return_json[res] = button_obj.return_json()

    for text_obj in list_of_text_objs:
        pprint(text_obj.is_nullified)
        if not text_obj.is_nullified:
            res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))
            main_return_json["data"][res] = text_obj.return_json()

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  
    imS = cv2.resize(image, (1600, 1000))                # Resize image


    # cv2.imshow('Contours', imS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pprint(main_return_json)

    return main_return_json
    
