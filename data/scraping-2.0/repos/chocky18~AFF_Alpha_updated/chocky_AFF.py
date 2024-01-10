from sys import argv, exit, platform
import openai
import time
from playwright.sync_api import sync_playwright
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow,QAction, QInputDialog,QFileDialog, QPushButton, QVBoxLayout, QWidget,QTextEdit, QLabel,QLineEdit
from PyQt5.QtGui import QIcon,QPixmap,QFont
from PyQt5.QtCore import QSize,Qt,QPoint,QTimer,QRect
import sys
import json
import re
import playwright
import tiktoken
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pdfx

urlList=[]
urlIndex=0
previous_command_list = ""
input_data={}
fileID=[]
fileLoaded=False
file_path=None
file_path2=None

drop_down_prompt_template ="""
You have given drop_down_values which are option IDs with Values (option : <option_id>, Value: <option_value>) and a Sentence or word
Find the most similar value in a list of dictionaries to the given word or sentence
Find the synonym for the given sentence
You have to understand the Values in option and find which value in option is more similar to the Sentence or word

If you guess the which option ID with value is closer to the word, return me the value and name of option data
{"key":value}

format of data options:
[{"option_value": <option_id>,"option_value": <option_id>,...}]
option : <option_id>, Value: <option_value>

Previous_Answer:
A word or sentence

Here are some examples:

EXAMPLE 1:
==================================================
[{"Yes":67,"No":69}]
Previous_Answer:  No, we do not have revenue yet. We are in the process of training a multi-modal system which we think is an opening in the market.
------------------
Previous_Answer: Previous_Answer
data: data
YOUR ANSWER: {'No':69}

EXAMPLE 2:
==================================================

[{"What are you looking for?":31,"Virtual Membership":32,"Full time membership":33,"Hot desk membership":34,"Events":35,"Programmes":36,"Corporate partnership":37}]
Previous_Answer:  We invite you to join us on this exciting journey as we continue to refine and enhance Workplete, ultimately bringing it to the masses.
------------------
Previous_Answer: Previous_Answer
data: data
YOUR ANSWER: {'Full time membership':33}
=================================================
---
MAP THE DATA:
data: $data
Previous_Answer: $Previous_Answer
Your Answer:
"""

prompt_template = """
Given a simplified HTML representation of a webpage, your task is to identify the appropriate mappings between input IDs and names, as well as text IDs and names within the HTML elements.

Instructions:
1. Map the input ID with the corresponding text name. For example, if an input has the ID "25" and a text element has the value "First Name", map them together.
2. Only map input IDs with text values. Do not include text IDs without corresponding input IDs in the mapping.
3. If there is a button with the name "Start", map it to the corresponding button ID and value. For instance, if a button has the ID "5" and the value "Start", map them together as "Start": 5.
4. If a button named "OK" immediately follows an input ID, map it to the input ID. For example, if a button has the ID "7" and the value "OK", map them together as 7: "OK".
5. Do not map text IDs to other text IDs; only map text names to input IDs.
6. Exclude text names that do not have a corresponding input ID from the mapping.
7. If there are options in a sequence, include them as a list of dictionaries. Each dictionary should have a key-value pair where the key is the option value and the value is the option ID.
   For example, if the question is "How Did You Hear About Dreamit?" and the options are "Dreamit Team Member", "Internet Search", and "Investor or VC", the mapping should be:
   "How Did You Hear About Dreamit?": [{'Dreamit Team Member': 48}, {'Internet Search': 49}, {'Investor or VC': 50}]

Data Format:

input: id="<input_id>" type="text" name="<answer text">
text: id="<text_id" value="<question text">
button: id="<button_id" value="<button text">
checkbox: id="<checkbox_id" value="<checkbox text">
option: id="<option_id" value="<option text">

Here are some examples:


EXAMPLE 1:
==================================================
data:
text: id="0" value="First name"
input: id="1" type="text" name="firstname"
text: id="2" value="Last name"
input: id="3" type="text" name="lastname"
text: id="4" value="Company name"
input: id="5" type="text" name="company"
text: id="6" value="City"
input: id="7" type="text" name="city"
text: id="48" value="Sector*"
select: id="49" name="EditableTextField_55b96"
option: id="50" value="Please select a sector"
option: id="51" value="Aerospace and Defence"
option: id="52" value="Automotive"
option: id="53" value="Charities"
option: id="54" value="Construction"
option: id="55" value="Creative"
option: id="56" value="Digital"
option: id="57" value="Education"
option: id="58" value="Electronics"
option: id="59" value="Energy"
option: id="60" value="Engineering"


------------------
YOUR ANSWER: {"First name":1, "Last name":3, "Company name":5,"City":7,"Sector":[{'Please select a sector':50,'Aerospace and Defence':51,'Automotive':52,'Charities':53,'Construction':54,'Creative':55,'Digital':56,'Education':57,'Electronics':58,'Engineering':59,'Energy':60}]}


==================================================

EXAMPLE 2:
==================================================
data:
text: id="79" value="Accelerator*"
select: id="80" name="EditableTextField_94bc9"
option: id="81" value="Please select a programme"
option: id="82" value="Academic Accelerator"
option: id="83" value="Big Ideas Programme - Perth"
option: id="84" value="Bitesize Business Series"
option: id="85" value="Build, Run and Scale - Perth & Ki

------------------
YOUR ANSWER: {"Accelerator":[{'Please select a programme':81,'Academic Accelerator':82,'Big Ideas Programme - Perth':83,'Bitesize Business Series':84,'Build, Run and Scale - Perth & Ki':85}]}
==================================================

EXAMPLE 3:
==================================================
data:
button: id="7" type="submit" value="NEXT"
text: id="41" placeholder="Enter your How Did You Hear About Dreamit?"
text: id="42" value="How Did You Hear About Dreamit?"
select: id="43" name="how_did_you_hear_about_dreamit_"
option: id="44" value="Please Select"
option: id="45" value="Accelerator Rankings"
option: id="46" value="Angel List"
option: id="47" value="Conference or Event"
option: id="48" value="Dreamit Alumni"
option: id="49" value="Dreamit Team Member"
option: id="50" value="Internet Search"
option: id="51" value="Investor or VC"
option: id="52" value="Media Story"
option: id="53" value="Social Media"
option: id="54" value="Video / Youtube"
text: id="55" placeholder="Enter your Select Your Vertical"
text: id="56" value="Select Your Vertical"
text: id="57" value="*"
text: id="58" value="HealthtechSecuretech"
text: id="59" value="Healthtech"
radio: id="60" value="Dreamit HealthTech" type="radio" name="which_dreamit_program_are_you_applying_to_"       
text: id="61" value="Healthtech"
text: id="62" value="Securetech"
radio: id="63" value="Dreamit SecureTech" type="radio" name="which_dreamit_program_are_you_applying_to_"       
text: id="64" value="Securetech"
button: id="65" value="Submit" type="submit"
------------------
YOUR ANSWER: {"NEXT":7,"How Did You Hear About Dreamit?":[{'Please Select':44,'Accelerator Rankings':45,'Angel List':46,'Conference or Event':47,'Dreamit Alumni':48,'Dreamit Team Member':49,'Internet Search':50,'Investor or VC':51,'Media Story':52,'Social Media':53,'Video / Youtube':54}],"Select Your Vertical":[{'Dreamit HealthTech':60,'Dreamit SecureTech':63}]}


data:$data
YOUR ANSWER:
"""

Text_summarization_prompt_template ="""

Summarize the given text into a concise answer. If the text already has the meaning of not containing specific information then it must return "I don't know". 

Text: <input_text>

Your Answer:

EXAMPLE 1:
==================================================
Text Summarize it:
Text: "We are open to investments and are actively seeking additional investments beyond the $250k SAFE from Nat & Daniel to help us achieve our product vision."
YOUR ANSWER: We Looking for Investments

EXAMPLE 2:
==================================================
Text Summarize it:
Text: "I don't have any information."
YOUR ANSWER: I don't Know


EXAMPLE 3:
==================================================
Text Summarize it:
Text: "My email address is chockynaresh18@gmail.com"
YOUR ANSWER: chockynaresh18@gmail.com
=================================================
---
Text Summarize it:
Text: <input_text>
Your Answer:
"""
class Crawler:
    def __init__(self):
        self.browser = (
            sync_playwright()
            .start()
            .chromium.launch(
                headless=False,
            )
        )

        self.page = self.browser.new_page()


    def get_element_attributes(self,element_info):
        tag = element_info.get("tag")
        attrs = element_info.get("attributes", {})
        value = element_info.get("value", "")
        innerText = element_info.get("innerText", "")
        xpath = element_info.get("xpath", "")

        attr_dict = {}

        attr_dict = {attr: attrs[attr] for attr in ['value','type', 'title', 'placeholder', 'name', 'aria-label', 'role'] if attr in attrs and attrs[attr]}

        if tag in ['button', 'textarea', 'option'] or attrs.get('role') in ['button', 'checkbox', 'radio'] or attr_dict.get('type') in ['submit', 'checkbox', 'radio']:
            if value is None or value == "":
                value = innerText
            if value is not None and value != "":
                attr_dict['value'] = value
        elif (tag == 'input' and attrs.get('type') != 'submit') or attr_dict.get('role') == 'textbox':
            value = attrs.get('value')
            if value is not None and value != "":
                attr_dict['value'] = value
        else:
            if value is not None and value.strip():
                attr_dict['value'] = value
        return xpath, attr_dict

    def crawl(self):
        script = '''
            (function() {
                function getXPath(element) {
                    if (element === null) {
                        return '';
                    }
                    if (element === document.documentElement) {
                        return '/html';
                    } else if (element === document.body) {
                        return '/html' + '/body';
                    } else {
                        let ix = 0;
                        let siblings = element.parentNode.childNodes;
                        for (let i = 0; i < siblings.length; i++) {
                            let sibling = siblings[i];
                            if (sibling === element) {
                                return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                                ix++;
                            }
                        }
                    }
                }

                function getElementInfo(element) {
                    let rect = element.getBoundingClientRect();
                    if (rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth)) {
                        let attributes = {};
                        for (let attr of element.attributes) {
                            attributes[attr.name] = attr.value;
                        }

                        let xpath = getXPath(element);
                    let parentElement = element.parentNode;
                        let parentRect = parentElement.getBoundingClientRect();

                        let isVisible = rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth);
                        let isParentVisible = parentRect.top >= 0 && parentRect.left >= 0 && parentRect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && parentRect.right <= (window.innerWidth || document.documentElement.clientWidth);

                        return {
                            tag: element.tagName.toLowerCase(),
                            attributes: attributes,
                            value: getDirectTextContent(element),
                            innerText: element.textContent.trim(),
                            xpath: xpath,
                            isVisible: isVisible && isParentVisible
                        };
                    }
                    return null;
                }

                function getDirectTextContent(element) {
                    let text = '';
                    for (let node of element.childNodes) {
                        if (node.nodeType === 3) {
                            text += node.textContent;
                        }
                    }
                    return text.trim();
                }

                function isElementJavaScript(el) {
                    return el.tagName.toLowerCase() === 'script' || el.textContent.includes('function(');
                }

                Array.from(document.querySelectorAll('a[target="_blank"]')).forEach(
                    link => link.removeAttribute('target')
                );

                let elements = document.querySelectorAll('body *');
                let result = [];

                for (let element of elements) {
                    let computedStyle = window.getComputedStyle(element);
                    if ((element.offsetWidth > 0 || element.tagName.toLowerCase() === 'option') && computedStyle.display !== 'none' && !isElementJavaScript(element)) {
                        let info = getElementInfo(element);
                        if (info) {
                            result.push(info);
                        }
                    }
                }

                return result;
            })();
        '''

        
        def extract_elements(script, target_page, iframe_id=None):
            elements_in_viewport = target_page.evaluate(script)

            for element_info in elements_in_viewport:
                xpath, attr_dict = self.get_element_attributes(element_info)
                display = element_info.get("computedStyle", {}).get("display")
                isVisible = element_info.get("isVisible", True)
                if attr_dict and display != "none" and isVisible:
                    attr_dict["id"] = id_counter[0]
                    tag = element_info.get("tag")

                    # Construct a string for output
                    attr_str = ' '.join([f'id="{attr_dict["id"]}"'] + [f'{key}="{value}"' for key, value in attr_dict.items() if key not in ['id', 'role']])
                    if tag in ['button'] or attr_dict.get('role') in ['button'] or attr_dict.get('type') in ['submit']:
                        output_list.append(f"button: {attr_str}")
                    elif tag in ['select']:
                            output_list.append(f"select: {attr_str}")
                    elif tag in ['option']:
                        output_list.append(f"option: {attr_str}")
                    elif(tag=='input' and attr_dict.get('type')=='radio'):
                        output_list.append(f"radio: {attr_str}")
                    elif(tag=='input' and attr_dict.get('type')=='checkbox'):
                        output_list.append(f"checkbox: {attr_str}")
                    elif (tag == 'input' and attr_dict.get('type') != 'submit') or tag == 'textarea' or attr_dict.get('role') == 'textbox':
                        output_list.append(f"input: {attr_str}")
                    # elif tag == 'a' or attr_dict.get('role') in ['link']:
                        # output_list.append(f"link: {attr_str}")
                    else:
                        output_list.append(f"text: {attr_str}")

                    if iframe_id!=None:
                        xpath = f"{iframe_id}{xpath}"
                    xpath_dict[xpath] = attr_dict

                    id_counter[0] += 1

                    # If the element is an iframe, handle its content immediately
                    if tag == 'iframe':
                        iframe_element = target_page.query_selector(f'xpath={xpath}')
                        if iframe_element:
                            iframe_content = iframe_element.content_frame()
                            iframes_list.append((attr_dict["id"], iframe_content))  # Store the iframe and its ID in the list
                            extract_elements(script, iframe_content, attr_dict["id"])

    
        id_counter = [0]
        xpath_dict = {}
        output_list = []
        iframes_list = []

        # Extract elements in the main page
        extract_elements(script, self.page)

        # print(f"\ntotal no. of elements: {id_counter[0]}")
        return output_list, xpath_dict,iframes_list
    

    def get_xpath_by_id(self,id,xpath_dict):
        for xpath, attrs in xpath_dict.items():
            if attrs.get('id') == id:
                return xpath 
        return None

    def get_iframe_by_xpath(self,xpath, iframes_list):
        iframe_id = xpath.split("/")[0]  # extract the iframe_id from the xpath
        if iframe_id:  # check if iframe_id is not an empty string
            for id, frame in iframes_list:
                if id == int(iframe_id):
                    return frame
        return None

    def click_element(self,id, xpath_dict, iframes_list):
        try:
            xpath = self.get_xpath_by_id(id, xpath_dict)
            frame = self.get_iframe_by_xpath(xpath, iframes_list)
            if frame:
                # remove the iframe_id from the xpath
                xpath = re.sub(r'^\d+/', '/', xpath)
                if xpath.split('/')[-1].startswith('option'):
                    # If the element is an option, get its parent select element and select the option
                    select_xpath = '/'.join(xpath.split('/')[:-1])
                    select_element = frame.query_selector(f'xpath={select_xpath}')
                    option_element = frame.query_selector(f'xpath={xpath}')
                    value = option_element.get_attribute('value')
                    select_element.select_option(value)
                else:
                    frame.click(f'xpath={xpath}')  
            else:
                if xpath.split('/')[-1].startswith('option'):
                    # If the element is an option, get its parent select element and select the option
                    select_xpath = '/'.join(xpath.split('/')[:-1])
                    select_element = self.page.query_selector(f'xpath={select_xpath}')
                    option_element = self.page.query_selector(f'xpath={xpath}')
                    value = option_element.get_attribute('value')
                    select_element.select_option(value)
                else:
                    self.page.click(f'xpath={xpath}')
        except:
            _crawler.scroll_down()
            print("Error in click_element function")

    def type_into_element(self,id, xpath_dict, iframes_list, text):
        xpath = self.get_xpath_by_id(id, xpath_dict)
        frame = self.get_iframe_by_xpath(xpath,iframes_list)
        if frame:
            # remove the iframe_id from the xpath
            xpath = re.sub(r'^\d+/', '/', xpath)
            frame.fill(f'xpath={xpath}',text)   
        else:
            self.page.fill(f'xpath={xpath}', text)

    def type_and_submit(self,xpath_dict, iframes_list,id, text):
        xpath = self.get_xpath_by_id(id, xpath_dict)
        frame = self.get_iframe_by_xpath(xpath,iframes_list)
        if frame:
            # remove the iframe_id from the xpath
            xpath = re.sub(r'^\d+/', '/', xpath)
            frame.fill(f'xpath={xpath}',text)   
            frame.press(f'xpath={xpath}','Enter')
        else:
            self.page.fill(f'xpath={xpath}', text)
            self.page.press(f'xpath={xpath}','Enter')

    def scroll_up(self):
        current_scroll_position = self.page.evaluate('window.pageYOffset')
        viewport_height = self.page.viewport_size['height']
        new_scroll_position = max(current_scroll_position - viewport_height, 0)
        self.page.evaluate(f'window.scrollTo(0, {new_scroll_position})')

    def scroll_down(self):
        current_scroll_position = self.page.evaluate('window.pageYOffset')
        viewport_height = self.page.viewport_size['height']
        scroll_amount = 100  # Change this value to set the scroll amount

        new_scroll_position = current_scroll_position + viewport_height - scroll_amount
        self.page.evaluate(f'window.scrollTo(0, {new_scroll_position})')

    # def scroll_down(self):
    #     current_scroll_position = self.page.evaluate('window.pageYOffset')
    #     viewport_height = self.page.viewport_size['height']
    #     new_scroll_position = current_scroll_position + viewport_height
    #     self.page.evaluate(f'window.scrollTo(0, {new_scroll_position})')

    def goToURL(self,url):
        response = self.page.goto(url=url, timeout=0)
        self.page.wait_for_load_state()
        status = response.status if response else "unknown"
        print(f"Navigating to {url} returned status code {status}")


    def goPageBack(self):
        response = self.page.go_back(timeout=60000)
        self.page.wait_for_load_state()
        if response:
            print(
                f"Navigated back to the previous page with URL '{response.url}'."
                f" Status code {response.status}"
            )
        else:
            print("Unable to navigate back; no previous page in the history")


if (
__name__ == "__main__"
):
    while True:
        _crawler = Crawler()
        openai.api_key ="your openai key"

        import os 
        os.environ["OPENAI_API_KEY"] = "your openai key"
        import re


        def print_help():
            print(
                "(g) to visit url\n(u) scroll up\n(d) scroll down\n(c) to click\n(t) to type\n" +
                "(h) to view commands again\n(r/enter) to run suggested command\n(o) change objective"
            )


        def num_tokens_from_string(string: str, encoding_name: str) -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.encoding_for_model(encoding_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        
        def get_gpt_command(string_data):
            prompt = prompt_template
            prompt = prompt.replace("$data", string_data)
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "YOUR ANSWER: "}])

            input_string = response["choices"][0]["message"]["content"]
            return input_string


        def gpt_for_drop_down(optiondata ,Previous_Answer):
            prompt = drop_down_prompt_template
            prompt = prompt.replace("$Previous_Answer", Previous_Answer)
            prompt = prompt.replace("$data", str(optiondata))
            print("options_in_gpt_command",optiondata)
            print("Previous answer", Previous_Answer)
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "YOUR ANSWER: "}])

            input_string = response["choices"][0]["message"]["content"]
            return input_string
        
        def gpt_for_text_summarization(Text):
            prompt = Text_summarization_prompt_template
            prompt = prompt.replace("<input_text>", Text)
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "YOUR ANSWER: "}])

            input_string = response["choices"][0]["message"]["content"]
            return input_string
        
        class CustomMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Workplete")
                self.setWindowFlag(Qt.WindowStaysOnTopHint)
                self.setWindowFlag(Qt.FramelessWindowHint)
                self.setFixedSize(400, 200)
                self.draggable = False
                self.dragging = False
                self.drag_start_position = None
                self.offset = QPoint()
                self.resize_handle_size=20 

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.drag_start_position = event.globalPos()

            def mouseMoveEvent(self, event):
                if self.dragging:
                    delta = event.globalPos() - self.drag_start_position

                    if (
                        event.pos().x() < self.resize_handle_size
                        and event.pos().y() < self.resize_handle_size
                    ):
                        self.resize(self.width() - delta.x(),
                                    self.height() - delta.y())
                        self.move(self.x() + delta.x(), self.y() + delta.y())
                    elif (
                        event.pos().x() > self.width() - self.resize_handle_size
                        and event.pos().y() < self.resize_handle_size
                    ):
                        self.resize(self.width() + delta.x(),
                                    self.height() - delta.y())
                        self.move(self.x(), self.y() + delta.y())
                    elif (
                        event.pos().x() < self.resize_handle_size
                        and event.pos().y() > self.height() - self.resize_handle_size
                    ):
                        self.resize(self.width() - delta.x(),
                                    self.height() + delta.y())
                        self.move(self.x() + delta.x(), self.y())
                    elif (
                        event.pos().x() > self.width() - self.resize_handle_size
                        and event.pos().y() > self.height() - self.resize_handle_size
                    ):
                        self.resize(self.width() + delta.x(),
                                    self.height() + delta.y())
                    else:
                        self.move(self.x() + delta.x(), self.y() + delta.y())

                    self.drag_start_position = event.globalPos()

            def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.dragging = False

            def getResizeHandleAt(self, pos):
                handle_size = self.resize_handle_size
                rect = self.rect()

                if QRect(0, 0, handle_size, handle_size).contains(pos):
                    return "TopLeft"
                elif QRect(rect.width() - handle_size, 0, handle_size, handle_size).contains(pos):
                    return "TopRight"
                elif QRect(0, rect.height() - handle_size, handle_size, handle_size).contains(pos):
                    return "BottomLeft"
                elif QRect(rect.width() - handle_size, rect.height() - handle_size, handle_size, handle_size).contains(pos):
                    return "BottomRight"
                elif QRect(0, handle_size, handle_size, rect.height() - 2 * handle_size).contains(pos):
                    return "Left"
                elif QRect(rect.width() - handle_size, handle_size, handle_size, rect.height() - 2 * handle_size).contains(pos):
                    return "Right"
                elif QRect(handle_size, 0, rect.width() - 2 * handle_size, handle_size).contains(pos):
                    return "Top"
                elif QRect(handle_size, rect.height() - handle_size, rect.width() - 2 * handle_size, handle_size).contains(pos):
                    return "Bottom"

                return None

            def getResizeCursor(self, handle):
                if handle in ["TopLeft", "BottomRight"]:
                    return Qt.SizeFDiagCursor
                elif handle in ["TopRight", "BottomLeft"]:
                    return Qt.SizeBDiagCursor
                elif handle in ["Left", "Right"]:
                    return Qt.SizeHorCursor
                elif handle in ["Top", "Bottom"]:
                    return Qt.SizeVerCursor

                return Qt.ArrowCursor

            def resizeTop(self, pos):
                handle_size = self.resize_handle_size
                diff = self.mapToGlobal(QPoint(pos.x(), pos.y())) - \
                    self.mapToGlobal(QPoint(0, 0))
                new_height = self.height() - diff.y()
                if new_height >= self.minimumHeight():
                    self.setGeometry(self.x(), self.y() + diff.y(),
                                    self.width(), new_height)

            def resizeBottom(self, pos):
                handle_size = self.resize_handle_size
                diff = self.mapToGlobal(QPoint(pos.x(), pos.y())) - \
                    self.mapToGlobal(QPoint(0, 0))
                new_height = self.height() + diff.y()
                if new_height >= self.minimumHeight():
                    self.resize(self.width(), new_height)

            def resizeLeft(self, pos):
                handle_size = self.resize_handle_size
                diff = self.mapToGlobal(QPoint(pos.x(), pos.y())) - \
                    self.mapToGlobal(QPoint(0, 0))
                new_width = self.width() - diff.x()
                if new_width >= self.minimumWidth():
                    self.setGeometry(self.x() + diff.x(), self.y(),
                                    new_width, self.height())

            def resizeRight(self, pos):
                handle_size = self.resize_handle_size
                diff = self.mapToGlobal(QPoint(pos.x(), pos.y())) - \
                    self.mapToGlobal(QPoint(0, 0))
                new_width = self.width() + diff.x()
                if new_width >= self.minimumWidth():
                    self.resize(new_width, self.height())

            def is_resizable_area(self, pos):
                width = self.width()
                height = self.height()
                return (
                    pos.x() <= self.resize_handle_size
                    or pos.x() >= width - self.resize_handle_size
                    or pos.y() <= self.resize_handle_size
                    or pos.y() >= height - self.resize_handle_size
                )

            def get_resize_direction(self, pos):
                width = self.width()
                height = self.height()
                if pos.x() <= self.resize_handle_size and pos.y() <= self.resize_handle_size:
                    return "topleft"
                elif pos.x() >= width - self.resize_handle_size and pos.y() <= self.resize_handle_size:
                    return "topright"
                elif pos.x() <= self.resize_handle_size and pos.y() >= height - self.resize_handle_size:
                    return "bottomleft"
                elif pos.x() >= width - self.resize_handle_size and pos.y() >= height - self.resize_handle_size:
                    return "bottomright"
                elif pos.x() <= self.resize_handle_size:
                    return "left"
                elif pos.x() >= width - self.resize_handle_size:
                    return "right"
                elif pos.y() <= self.resize_handle_size:
                    return "top"
                elif pos.y() >= height - self.resize_handle_size:
                    return "bottom"
                else:
                    return None


        def create_window():
            def load_file():
                global fileLoaded,file_path,file_path2
                file_dialog = QFileDialog()
                file_path, _ = file_dialog.getOpenFileName(window, 'Select File', '', 'Text Files (*.txt);;PDF Files (*.pdf);;DOC Files (*.doc *.docx);;CSV Files (*.csv)')
                if file_path:
                    file_extension=os.path.splitext(file_path)[1]
                    file_extension=file_extension.lower()

                    if(file_extension==".pdf"):
                        loader = PyPDFLoader(file_path)
                        file = loader.load()
                    elif(file_extension in ('.doc','.docx')):
                        loader=Docx2txtLoader(file_path)
                        file=loader.load()
                    elif(file_extension == ".txt"):
                        loader=TextLoader(file_path)
                        file=loader.load()
                    elif(file_extension=='.csv'):
                        loader=CSVLoader(file_path)
                        file=loader.load()
                    
                    try:
                        documents=[]
                        # Split the documents into chunks
                        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        texts = text_splitter.split_documents(file)
                        
                        documents.extend(texts)
                        try:
                            loader=TextLoader("temp.txt")
                            file=loader.load()
                            documents.extend(file)
                        except:
                            print("There is no temp.txt file")

                        # Select which embeddings to use
                        embeddings = OpenAIEmbeddings()
                        
                        # Create the VectorStore to use as the index
                        db = Chroma.from_documents(documents, embeddings)
                        
                        # Expose this index in a retriever interface
                        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                        
                        # Create a chain to answer questions
                        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
                        fileLoaded=True
                        file_path2=file_path
                        file_path=qa
                        return qa
                    except Exception as e:
                        print(f"Error: Failed to load the file - {e}")
                        return None
                else:
                    # Handle the case when the user cancels file selection
                    return None

            def pdfInputAddition(sentence):
                with open("temp.txt","a") as file:
                    file.write(sentence)

            def pdf_extract():
                global urlList
                file_dialog = QFileDialog()
                file_path, _ = file_dialog.getOpenFileName(window, 'Select File', '', 'PDF Files (*.pdf);;DOC Files (*.doc *.docx);;CSV Files (*.csv)')
                pdf=pdfx.PDFx(file_path)
                print(pdf.get_references_as_dict())
                urlList=pdf.get_references_as_dict()['url']
                url_input.clear()
                url_input.setText(urlList[urlIndex])
                on_submit_clicked()
            
            def pdfCall():
                global urlIndex,urlList
                print(urlList,urlIndex,"Inside pdfCall")
                urlIndex+=1
                url_input.clear()
                try: 
                    url_input.setText(urlList[urlIndex])
                    on_submit_clicked()
                except:
                    print("Filled a single form")
                    sys.exit(0)
                    

                       
            def on_submit_clicked():
                global fileLoaded,file_path
                url = url_input.text()  # Get the URL from the input box
                # url = url_input.text()  # Get the URL from the input box
                if url:
                    try:
                        _crawler.goToURL(url)
                    except Exception as e:
                        print(f"Error loading URL: {e}")
                        pdfCall()
                
                if not fileLoaded:
                    print("Loading...")
                    file_path = load_file()
                if file_path:
                    # Execute functions that require the URL and file
                    gpt_cmd = ""
                    while True:
                        start = time.time()
                        visibledom, xpath_dict,iframes_list = _crawler.crawl()
                        print(iframes_list)
                        xpath_dict = {k: v for k, v in xpath_dict.items() if v is not None}                        
                        string_text = "\n".join(visibledom)
                        print("string_text", string_text)
                        gpt_cmd = get_gpt_command(string_text)
                        print("gpt command: ", gpt_cmd)
                        gpt_cmd = gpt_cmd.strip()
                        data = {}
                        if len(gpt_cmd) > 0:
                            try:
                                data = eval(gpt_cmd)
                            except Exception as e:
                                print(f"Error in evaluating gpt_cmd: {e}")
                                _crawler.scroll_down()
                            if all(key in data for key in ['PREV', 'Powered by Typeform']):
                                del data['Powered by Typeform']
                                data.pop('PREV', None)
                            
                            
                            swapped_data = {}

                            for key, value in data.items():
                                if isinstance(key, int):
                                    swapped_data[str(value)] = key
                                else:
                                    swapped_data[key] = value
                            previous_llmaanswer = ''
                            print("swapped_data",swapped_data)
                            keys_to_remove = []  # Create a list to store keys for removal
                            for key, value in swapped_data.items():

                                print("key",key)
                                
                                

                                click_keywords = ['NEXT','OK','Submit','submit']

                                if any(keyword in key for keyword in click_keywords):
                                    print("key in click ",key)
                                    print("value in click",value)
                                    _crawler.click_element(value, xpath_dict, iframes_list)
                                    # swapped_data.pop(key) 
                                    # print("swapped_data after click",swapped_data)
                                    continue
                                                                                                                                                             
                                
                                result = file_path({"query": key})
                                llmaanswer = result['result']
                                Text_summarized=gpt_for_text_summarization(llmaanswer)
                                print("llmaanswer",llmaanswer)
                                print("Text_summarized",Text_summarized)
                                sub_mappings = {}
                                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                                    optiondata_str = json.dumps(value)
                                    similarity_check = gpt_for_drop_down(optiondata_str,Text_summarized)
                                    print("similarity_check",similarity_check)
                                    if similarity_check is not None and 'None' not in similarity_check:
                                        data = eval(similarity_check)
                                        for key,value in data.items():
                                            if key in["I don't know"] :
                                                user_input, ok_pressed = QInputDialog.getText(window, "Popup Window", f"Enter input for {key} with optionIDs {value}: ")              
                                                if ok_pressed:
                                                    print("User input:", user_input)
                                                    llmaanswer = user_input 
                                                    _crawler.click_element(int(llmaanswer),xpath_dict,iframes_list)      
                                            _crawler.click_element(value,xpath_dict, iframes_list)
                                            if key.lower() in['Submit','submit','subscribe']:
                                                pdfCall()
                                    else: 
                                        user_input, ok_pressed = QInputDialog.getText(window, "Popup Window", f"Enter input for {key} with optionIDs {value}: ")              
                                        if ok_pressed:
                                            print("User input:", user_input)
                                            llmaanswer = user_input 
                                            _crawler.click_element(int(llmaanswer),xpath_dict,iframes_list)      
                                else:
                                    try: 
                                        keywords = ["don't know", "don't"]
                                        if any(keyword in Text_summarized for keyword in keywords):
                                            user_input, ok_pressed = QInputDialog.getText(window, "Popup Window", f"Enter input for {key}: ")              
                                            if ok_pressed:
                                                pdfInputAddition(f"{key}: {user_input}\n")
                                                print("User input:", user_input)
                                                Text_summarized=user_input  
                                        _crawler.type_into_element(value,xpath_dict, iframes_list,Text_summarized)
                                    except:
                                        # if key in ['OK']:
                                        #     _crawler.click_element(value,xpath_dict,iframes_list)

                         
                                        _crawler.click_element(value,xpath_dict,iframes_list)
                                        if key.lower() in['Submit','submit','subscribe']:
                                            pdfCall()
                
                            _crawler.scroll_down()
                            time.sleep(5)
                                        
            app = QApplication(sys.argv)
            
            # Create the main window
            window = CustomMainWindow()
            window.setStyleSheet("QMainWindow{background-color: black; font-weight: bold; border-radius: 10px;}")

            # window = QMainWindow()
            window.setWindowTitle("Workplete")
            window.setWindowFlag(Qt.WindowStaysOnTopHint)
            window.setWindowFlag(Qt.FramelessWindowHint)
            window.setFixedSize(400, 200)
            window.move(10, 550)

            # Create custom title bar
            title_bar = QWidget(window)
            title_bar.setGeometry(0, 0, 400, 40)
            title_bar.setStyleSheet("background-color: black; ")

            # Create logo
            logo_label = QLabel(window)
            logo_pixmap = QPixmap("logo_new.png").scaledToHeight(40)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setGeometry(8, -2, logo_pixmap.width(), logo_pixmap.height())

            # Create title label
            title_label = QLabel("Workplete", title_bar)
            title_label.setGeometry(70, -5, 200, 50)
            title_label.setStyleSheet("color: #ea9d59; font-size:20px; font-family:'Roboto-Mono';")

            # Create close button
            close_btn = QPushButton(title_bar)
            close_btn.setGeometry(350, -5, 50, 50)
            close_btn.setIcon(QIcon("close.png"))
            close_btn.setIconSize(QSize(20, 20))
            close_btn.setStyleSheet("QPushButton{background-color:#353b48; border:none;}")

            # Connect close button to close application
            close_btn.clicked.connect(window.close)

            or_label=QLabel("---OR---",window)
            or_label.setAlignment(Qt.AlignCenter)
            or_label.setStyleSheet("color: #ea9d59; font-size: 14px; font-family:'Roboto-Mono';")
            or_label.setGeometry(160,80,100,30)

            input_button = QPushButton('Select URL File', window)
            input_button.setGeometry(160, 120, 100, 30)
            input_button.clicked.connect(pdf_extract)
            input_button.setStyleSheet("QPushButton { background-color: #ea9d59; color: black; font-family: 'Roboto-Mono'; font-weight: bold;  border-radius: 10px;}")
            # input_button.clicked.connect(load_file)

            # Create the URL input box
            url_input = QLineEdit(window)
            url_input.setGeometry(10, 50, 380, 30)
            url_input.setStyleSheet("QLineEdit { background-color: black; border-radius: 15px; border: 1px solid #c8c8c8; font-size: 14px; font-family: 'Roboto Mono'; color: #ea9d59;}")
            url_input.setPlaceholderText("Enter URL")

            # Create the submit button
            submit_btn = QPushButton("Submit", window)
            submit_btn.setGeometry(300, 160, 80, 30)
            submit_btn.setStyleSheet("QPushButton { background-color: #ea9d59; color: black; font-weight: bold; font-family: 'Roboto-Mono'; border-radius: 10px;}")
            submit_btn.clicked.connect(on_submit_clicked)
            
            # Creeate the file input button for context file 
            context_btn=QPushButton("Context File",window)
            context_btn.setGeometry(20,160,80,30)
            context_btn.setStyleSheet("QPushButton { background-color: #ea9d59; color: black; font-weight: bold; font-family: 'Roboto-Mono'; border-radius: 10px;}")
            context_btn.clicked.connect(load_file)

            # Show the window
            window.show()
            sys.exit(app.exec_())
        _crawler.goToURL("https://www.google.com/")
        try:
            create_window()
        except KeyboardInterrupt:
            print("\n[!] Ctrl+C detected, exiting gracefully.")
            exit(0)
