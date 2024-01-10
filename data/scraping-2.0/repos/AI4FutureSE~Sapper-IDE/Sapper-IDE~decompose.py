import time
import openai
def gpt3(prompt, t, max_tokens):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=t,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=""
        )
        return response["choices"][0]["text"]

    except Exception as e:
        print(type(e), e)
        time.sleep(3)

decompose_template = """A user is interacting with a large language model. They are crafting prompts and giving them to the LLM in order to get the model to complete a task or generate output. 
 
In order for a large language model to better complete tasks or generate outputs, task requirements need to be decomposed. 
 
The decomposed subtasks can interact with a large language model like a human thought chain, reflecting data flow and control flow.
 
 
<Task Description>
I want to build a chatbot that will stop the conversation until someone says GoodBye.
</Task Description>
 
<Decomposed Subtasks>

<Control While User_Input not equal to GoodBye>

(Subtask1 Input:  Chat_History User_Input Bot_Response Output: Bot_Response Model LLM)
Combine chat history, user input, and bot response to get a new chat history.
 
(Subtask2 Input: Chat_History User_Input Output: Bot_Response Model LLM)
Keep the conversation going and prompt the user for more input

</Control>
</Decomposed Subtasks>
 
 
<Task Description>
I need to develop a function to obtain the weather conditions of the day according to the weather API and automatically draw 500x500 pixel RGB color paintings that meet the weather conditions, draw abstract paintings when the weather is rainy, and draw natural landscape paintings when the weather is sunny, so as to improve the user experience and entertainment.
</Task Description>
 
<Decomposed Subtasks >

(Subtask1 Input: None Output: Weather_Data Model OpenWeatherMap)
obtain weather conditions for the day
 
<Control If Weather equal to rainy>

(Subtask2 Input: Weather_Data Output: Painting_Description Model LLM)
generate descriptions of abstract paintings through weather information.

</Control>
 
<Control If Weather equal to sunny>

(Subtask3 Input: Weather_Data Output: Painting_Description Model LLM)
generate natural landscape descriptions of abstract paintings through weather information.

</Control>
 
(Subtask4 Input: Painting_Description Output: Painting; Model Image-generation-model)
Generate 500x500 pixel paintings according to Painting_Description.

</Decomposed Subtasks>
 
 
<Task Description>
{{Description}}
</Task Description>
Note: Output cannot have the same name
"""

decompose_template1 = """<Requirement Description>
I need to develop a function to obtain the weather conditions of the day according to the weather API and automatically draw 500x500 pixel RGB color paintings that meet the weather conditions, draw abstract paintings when the weather is rainy, and draw natural landscape paintings when the weather is sunny, so as to improve the user experience and entertainment.
</Requirement Description>

<Decomposed steps>
To achieve this function, you can follow the following steps to analyze and design the process:

(Step1 Input: None Output: API_Interface_Selection)
First, you need to choose an API interface used to obtain weather information. You can select some open weather APIs, such as OpenWeatherMap, AccuWeather, etc.

(Step2 Input: API_Interface_Selection Output: API_Key)
After selecting the API interface, you need to configure the API key to access the API. Generally, the API provider will provide a key, which can be found in the API documentation.

(Step3 Input: API_Key Output: Weather_Data)
Use the API key to access the API to get the weather data of the day. The data format is usually JSON or XML.

(Step4 Input: Weather_Data Output: Parsed_Data_Structure)
Parse the obtained JSON or XML format data into easy-to-handle data structures, such as dictionaries or objects.

(Step5 Input: Parsed_Data_Structure Output: Weather_Type)
Judge the weather type of the day according to the description information in the weather data. It can be classified according to weather conditions, such as sunny, cloudy and rainy days.

(Step6 Input: Weather_Type Output: RGB_Color_Value)
In order to generate paintings that meet weather conditions, you need to map the weather type to the corresponding RGB color value. You can map sunny days to blue tones, rainy days to gray tones, and snowy days to white tones.

(Step7 Input: Weather_Type&RGB_Color_Value Output: Painting)
Generate 500x500 pixel paintings according to weather type and corresponding RGB color values. For rainy days, you can generate abstract paintings, and for sunny days, you can generate natural landscape paintings.

(Step8 Input: Painting Output: Display_Painting)
Display the generated paintings to users to improve user experience and entertainment.
</Decomposed steps>

<Requirement Description>
{{Description}}
</Requirement Description>

<Decomposed steps>
"""

def decompose(query):
    decomposed_steps = gpt3(decompose_template.replace("{{Description}}", query), 0, 2048)
    return decomposed_steps

def Generasteps(query , OpenAIKey):
    openai.api_key = OpenAIKey
    # query = "I need to develop a function that allows users to search for nearby restaurants and hotels according to their current location and display the search results on the map."
    steps = decompose(query).split("\n\n")[1:]
    stepsJson = {}
    for j,step in enumerate(steps):
        if '(Subtask' in step:
            temp = step.split("\n")
            inp = temp[0].split(" ")[2].split("&")
            newinp = [[i, 'no'] for i in inp if i != 'None']
            # for i in inp:
            #     newinp.append([i,'no'])
            oup = temp[0].split(" ")[4]

            mod = temp[0].split(" ")[6][:-1]
            content = temp[1]
            js = {"content": content, "input": newinp, "output": [oup, 'no'], "model": mod}
            name = 'step' + str(j)
            stepsJson[name] = js

    return stepsJson

query = "I need to develop a function that allows users to search for nearby restaurants and hotels according to their current location and display the search results on the map."
query1 = "I need an automatic problem maker that can generate multiple choice math questions based on the difficulty and number of questions entered by the user."


pat = {"content": "How are you", "input": ["history", "chatbot"],"prompt":["prompt1","prompt2"] ,"output": "human", "model": "LLM"}
