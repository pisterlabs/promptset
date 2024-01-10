import os
import openai

# references
#
# 코드 설명 : 정답 코드에 대한 설명이 자동으로 이루어지는가?
# https://www.youtube.com/watch?v=q1qc-GcHGmY
# https://beta.openai.com/examples/default-explain-code

# libraries
#
# pip install openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "" #TODO::API_KEY

sample_class = """

class Log:
    def __init__(self, path):
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        f = open(path, "a+")

        # Check that the file is newline-terminated
        size = os.path.getsize(path)
        if size > 0:
            f.seek(size - 1)
            end = f.read(1)
            if end != "\n":
                f.write("\n")
        self.f = f
        self.path = path

    def log(self, event):
        event["_event_id"] = str(uuid.uuid4())
        json.dump(event, self.f)
        self.f.write("\n")

    def state(self):
        state = {"complete": set(), "last": None}
        for line in open(self.path):
            event = json.loads(line)
            if event["type"] == "submit" and event["success"]:
                state["complete"].add(event["id"])
                state["last"] = event
        return state
"""

sample_function = """

def solution(n):
    a,b = 1,1
    if n==1 or n==2:
        return 1
        
    for i in range(1,n):
        a,b = b, a+b

    return a
"""

sample_function2 = """

def bubble_sort(arr):
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                
"""

def get_explanation(input_code):
    function_prompt = "\n\"\"\"\nHere's what the above function is doing:\n1"
    prompt= input_code + function_prompt
    response = openai.Completion.create(
    model="code-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
    )    
    code_explanation = "1" + response["choices"][0]["text"]
    
    return code_explanation
if __name__=="__main__":
    
    class_prompt = "\n\"\"\"\nHere's what the above class is doing:\n1"
    function_prompt = "\n\"\"\"\nHere's what the above function is doing:\n1"
    #print(sample_class + class_prompt)
    print(sample_function2 + function_prompt)
    
    prompt= sample_function2 + function_prompt
    
    response = openai.Completion.create(
    model="code-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
    )    
    
    print(response)
    #print(response["choices"])
    print()
    print()
    print()
    print()
    code_explanation = "1" + response["choices"][0]["text"]
    print(code_explanation)