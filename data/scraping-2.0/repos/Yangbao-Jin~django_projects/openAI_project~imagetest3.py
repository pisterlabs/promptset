import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

import base64
import requests

# OpenAI API Key
#api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "pp2.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      { 
        "role": "system",
        "content": 
          {
            "type": "text",
            "text":"""你是计算机专家，这是一个读伪码的题,请识别图像中的计算机伪码，然后回答图片上的问题。
            Description of the ACSL Pseudo-code
We will use the following constructs in writing this code for this topic in ACSL:

Construct：	Code Segment
Operators：	! (not) , ^ or ↑(exponent), *, / (real division), % (modulus), +, -, >, <, >=, <=, !=, ==, && (and), || (or) in that order of precedence
Functions：	abs(x) - absolute value, sqrt(x) - square root, int(x) - greatest integer <= x
Variables：	Start with a letter, only letters and digits
Sequential statements：	
    INPUT variable
    variable = expression (assignment)
    OUTPUT variable
Decision statements：
    IF boolean expression THEN
        Statement(s)
    ELSE (optional)
        Statement(s)
    END IF
Indefinite Loop statements：	
    WHILE Boolean expression
        Statement(s)
    END WHILE
Definite Loop statements：	
    FOR variable = start TO end STEP increment
        Statement(s)
    NEXT
Arrays:	1 dimensional arrays use a single subscript such as A(5). 2 dimensional arrays use (row, col) order such as A(2,3). Arrays can start at location 0 for 1 dimensional arrays and location (0,0) for 2 dimensional arrays. Most ACSL past problems start with either A(1) or A(1,1). The size of the array will usually be specified in the problem statement.

Strings: Strings can contain 0 or more characters and the indexed position starts with 0 at the first character. An empty string has a length of 0. Errors occur if accessing a character that is in a negative position or equal to the length of the string or larger. The len(A) function will find the length of the string which is the total number of characters. Strings are identified with surrounding double quotes. Use [ ] for identifying the characters in a substring of a given string as follows:
S = “ACSL WDTPD” (S has a length of 10 and D is at location 9)

S[:3] = “ACS” (take the first 3 characters starting on the left)

S[4:] = “DTPD” (take the last 4 characters starting on the right)

S[2:6] = “SL WD” (take the characters starting at location 2 and ending at location 6)

S[0] = “A” (position 0 only).

String concatenation is accomplished using the + symbol

如果用户输入的伪码不符合这个语法，在解题之前，你需要提示用户输入的伪码不符合语法，并指出具体哪里违反了语法，比如第几行，第几列违反了语法。
            """
          },
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这是一个读伪码的题，请读取图片上的伪码，并请回答图片上的问题"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 2000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json=response.json()

content = response_json['choices'][0]['message']['content']
print(content)