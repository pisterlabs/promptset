import streamlit as st
from dotenv import load_dotenv
from time import time as now
import guidance
import re
import os
import sys
sys.path.append("/Users/allyne/Documents/GitHub/Unity-Agent/")

PLANNER_PROMPT = '''
{{#system~}}
You are an efficient, direct and helpful assistant tasked with helping shape a ubicomp space. 
Your role is to generate clear, precise, and effective instructions for altering a 3D space according to user requests.
When responding to user requests, you must closely follow past successful examples provided. 
Your instructions should replicate the steps from these examples as closely as possible, only deviating slightly if necessary to tailor the plan to the specific user request. 
{{~/system}}
{{#user~}}
As an assistant, create clear and precise instructions to alter a 3D ubicomp space according to user requests. 
                    
Follow these guidelines:
- Respond with a numbered set of instructions. 
- Your first instruction must be to either create an object or find the object the user is referring to.
-- For example, if the user uses phrases like "The table" and "This table", you should have an instruction like "Find a table in the user's field of view"
- Each instruction should modify only 1 property or behaviour.
- Properties that can be edited are: Position, Rotation, Size, Color, Illumination (Whether the object emanates light), Luminous Intensity (Brightness of the light between 1 and 10), Levitation (When an object is levitated, it floats). 
- If you need to edit the position of more than one object, include it within a single instruction. For example, use "Edit the Position property of each chair to be 0.5 meters in front of each room wall" instead of separate instructions for each chair.
- Your instructions must translate subjective terms into specific, measurable instructions. 
-- For example, terms like "big" or "close to me" can translate to “2 times its current size” and  “1m away from the user” respectively. Always cite explicit numbers.
-- Terms like "the table" or "this table" should translate to "table in the user's field of view"
- For colors, use RGBA values.
- Only instructions modifying the Position property can mention more than one object types. All other property modification can only mention ONE object type.

The space consists of 4 walls, 1 ceiling, and 1 floor.

You are limited to creating or modifying the following object types: (You must use the exact case-sensitive type of the object)
Chair, Fox, Lamp, LED Cube, Push Button, Table, Vase, Zombie

The user's prompt is {{task}}.

The format for response should strictly be:
    1. Instruction 1\n
    2. Instruction 2\n
    …

*Note: Output should not contain any text other than the instructions.*
{{~/user}}
{{#assistant~}}
{{gen "plan" max_tokens=2000 temperature=0}}
{{~/assistant}}
'''

FUNCTION_PROMPT = '''
{{#system~}}
You are an AI skilled in C# development for Unity's ubicomp space. 
You will assist in creating functions as part of a larger system. 
You will do so by translating pseudocode for a task to C# code.
{{~/system}}
{{#user~}}
Your work revolves around our proprietary C# system. Our system comprises of: 
- SceneAPI: This is the wrapper class for our ubicomp space. The methods here will allow for manipulating the whole space. 
- Object3D: Each object in the space is of this type, and the methods here are available for every object. The anchor for the position and rotation of each object is at the bottom center of the object. 
- Vector3D: Any 3-dimensional dataset uses this class to represent x, y, and z.
- Color3D: Color information using RGBA.
        
**As the script's class inherits from `SceneAPI`, you can directly call its methods without prefixing.**

Use the provided system to manipulate the room.
        
Follow these steps:
1. Write method(s) using C# code to implement the task.  
2. Declare private fields above your method(s) to track the object(s) you create and modify within the method. Ensure you always assign your object finding, creations, or modifications to these declared fields. If you are creating multiple objects, you can use a list to store them.
3. Use Debug.Log statements for action logging and error handling.
4. Adhere strictly to standard C# methods. 
5. Add comments to your code to explain your thought process.
        
Here are all the classes and functions you may use in your code:
```
Object types that can be created: (You must use the exact case-sensitive type of the object)
Chair, Fox, Lamp, LED Cube, Push Button, Table, Vase, Zombie 

namespace Enums
{
    public enum WallName
    {
        Left,
        Right,
        BackLeft,
        BackRight,
    }
}
public class Object3D
{
    public string GetType()
    public void WhenSelected(UnityAction<SelectEnterEventArgs> function)
    public void WhenNotSelected(UnityAction<SelectExitEventArgs> function)
    public void WhenHovered(UnityAction<HoverEnterEventArgs> function)
    public void WhenNotHovered(UnityAction<HoverExitEventArgs> function)
    public void SetColor(Color3D color)
    public void SetLuminousIntensity(float intensity)
    public void Illuminate(bool lit)
    public void SetPosition(Vector3D pos)
    public void SetRotation(Vector3D rot)
    public void SetSize(Vector3D s)
    public void SetSizeByScale(float s)
    public void Levitate(bool isLevitated)
    public bool IsLevitated()
    public Color GetColor()
    public float GetIntensity()
    public bool IsLit()
    public Vector3D GetPosition()
    public Vector3D GetRotation()
    public Vector3D GetSize()
    public GameObject ToGameObject()
}

public class SceneAPI
{
    public Vector3D GetWallPosition(WallName wallname)
    public List<Object3D> GetAllObject3DsInScene()
    public Object3D FindObject3DByName(string objName)
    // To find an object you could do 
    //  List<Object3D> objectsInView = GetAllObject3DsInFieldOfView();
    //  Object3D desiredObj = objectsInView.Find(obj => obj.GetType().Equals("ObjType"));
    public bool IsObject3DInFieldOfView(Object3D obj)
    public List<Object3D> GetAllObject3DsInFieldOfView() // This is good to get objects in the user's field of view. You can use this to find game objects referenced in the task. 
    public bool IsObjectTypeValid(string objectType)
    public List<string> GetAllValidObjectTypes()
    public Vector3D GetSceneSize()
    public Vector3D GetUserOrientation()
    public Vector3D GetUsersHeadPosition() // Not good for object creation and position movement methods, as the tall height will cause objects to topple
    public Vector3D GetUsersLeftHandPosition()
    public Vector3D GetUsersRightHandPosition()
    public Vector3D GetUsersLeftHandRotation()
    public Vector3D GetUsersRightHandRotation()
    public Vector3D GetUsersFeetPosition() //This is good for object creation and position movement methods
    public Object3D CreateObject(string newObjName, string objectType, Vector3D position, Vector3D rotation)
}

public class Vector3D
{
    // Vector3D cannot be used like Vector3. It only has the methods below. If you want to use Vector3 methods, convert it to Vector3 using ToVector3() first.
    public float x { get; set; }
    public float y { get; set; }
    public float z { get; set; }
    Vector3D(float x, float y, float z)
    public Vector3D ToVector3()
    public Vector3D FromVector3(Vector3 vec)
} // To add 2 vector 3Ds, use new Vector3D(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z) Same logic for subtract
//When using NavMeshAgent, make sure to convert all Vector3Ds to Vector3s using ToVector3() before using them in the NavMeshAgent methods.
// When performing calculations like multiplication, remember to convert all your Vector3Ds to Vector3s using ToVector3() before performing the calculations.
public class Color3D
{
    //All values below range from 0 to 1 
    public float r { get; set; }
    public float g { get; set; }
    public float b { get; set; }
    public float a { get; set; }
    Color3D(float r, float g, float b, float a)
}
```
The task to create a function for is: {{task}}. 
        
Your format for responses should strictly be: 
                    
// All class members should be declared here. 

private void Start(){
    // Insert method(s) that only need to be called once
}
                    
private void Update(){
    // If method(s) that need to be called repeatedly
}
                    
public void Method1()
{
    // Insert the method code here
}
// And so on... The number of methods will depend on the user's request. 

*Note: Output should not contain any text other than the class members and method(s). You must give the full code within the methods*
{{~/user}}
{{#assistant~}}
{{gen "function" temperature=0 max_tokens=4096}}
{{~/assistant}}
''' 

SCRIPT_PROMPT = '''
{{#system~}}
You are a skilled C# developer tasked with crafting a coherent C# script, ensuring that created objects and states are managed and utilized effectively across multiple methods.
{{~/system}}
{{#user~}}
User Task: {{task}}

Actionable Instructions to fulfil the user task: {{plan}}
                        
Methods for each step of the actionable plan: {{functions}}
        
Follow these steps:
1. Develop a public class inheriting from SceneAPI with a name relevant to the task context.
2. You must use all methods in your final script. The only method(s) and/or code you are permitted to remove are methods that repeat creating or finding object(s).
3. Integrate and modify the methods to maintain script coherence. 
4. Utilize class-level variables (e.g., private Object3D classMember;) to preserve state throughout the class.
5. Remove duplicate variables referencing the same object(s).
6. Ensure that the same object references are used in every method. Once an object is initialized, consistently use this reference in all related operations. 
7. For each method, do not use FindObject3DByName() or CreateObject() to re-assign the same object(s). Always check if there is the same object reference that was initialized in previous methods.
8. Use Debug.Log statements for action logging and error handling.
9. Adhere to standard C# methods and conventions and add comments to your code to explain your thought process.
10. All methods should be called under either Start() or Update().

Your format for responses should strictly be: 
```
public class YourChosenClassName : SceneAPI
{	
    // All class members and constants should be declared here. 
    // Remember to remove duplicate variables referencing the same object(s)
    // Remember to call the final correct variable names across all methods
    
    private void Start()
        {
            Method1();
            Method2();
            // And so on... The number of methods will depend on the user's request. 
        }
    private void Update()
        {
            Method3();
            // And so on... The number of methods will depend on the user's request.
        }
                    
    public void Method1()
        {
            // Insert the method code here
        }
    public void Method2()
        {
            // Insert the method code here
        }
    // And so on... The number of methods will depend on the user's request. 
}
```
*Note: Output should not contain any text other than script containing method(s). You must give the full code within the methods.*
{{~/user}}
{{#assistant~}}
{{gen "script" temperature=0 max_tokens=4096}}
{{~/assistant}}
'''

def get_class_name_from_code(code_string):
    # Extract the class name from the code string using regex
    match = re.search(r'public class (\w+)', code_string)
    if match:
        return match.group(1)
    return "generated_script" 

def create_and_download_cs_file(code_string):
    class_name = get_class_name_from_code(code_string)
    file_name = f"{class_name}.cs"
    file_path = f"no_memory_generated_scripts/{file_name}"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(code_string)
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="Download .cs file",
            data=file,
            file_name=file_name,
            mime='text/plain',
        )
    if btn:
        os.remove(file_name)

def clean_function_text(text):
    text = text.replace("```csharp\n", "")  # Replace the starting string with nothing
    text = text.replace("```", "")          # Replace the ending string with nothing
    return text.strip()

def generate_plan(task):
    guidance.llm = guidance.llms.OpenAI("gpt-4-0613")
    planner = guidance(PLANNER_PROMPT)
    resp = planner(task=task)
    return(resp["plan"])

def generate_function(task):
    guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-1106")
    coder = guidance(FUNCTION_PROMPT)
    resp = coder(task=task)
    return(resp["function"])

def edit_code_string(code_string):
    try:
        first_index = code_string.index("public class")
        last_index = code_string.rindex("}")  
        return code_string[first_index:last_index+1]  
    except ValueError:
        print("Invalid code: 'using' or '}' not found.")

def generate_script(task, plan, functions):
    guidance.llm = guidance.llms.OpenAI("gpt-4-0613")
    script = guidance(SCRIPT_PROMPT)
    resp = script(task=task, plan=plan, functions=functions)
    script = edit_code_string(resp["script"])
    script = "using UnityEngine;\nusing UnityEngine.Events;\nusing UnityEngine.XR.Interaction.Toolkit;\nusing System;\nusing System.Collections.Generic;\nusing Enums;\nusing UnityEngine.AI;\nusing System.Linq;\n\n" + script
    return(script)

def no_memory_test(task):
    st.markdown(f"## Received your task to generate a script for: {task}")
    st.markdown("## Generating the plan...")
    plan = generate_plan(task)
    st.write(plan)
    plans = plan.split("\n")
    plans = [plan for plan in plans if plan.strip()]
    functions = []
    st.markdown("## Generating functions...")
    for plan in plans:
        st.write("Generating function for \n ```" + plan + "```")
        function = generate_function(plan)
        function = clean_function_text(function)
        functions.append(function)
        st.write("\n```csharp\n" + function + "\n\n")
    st.markdown("## Generating the entire script...")
    script = generate_script(task, plan, functions)
    st.write("```csharp\n" + script)
    st.write("\n\nDownload the script here:")
    create_and_download_cs_file(script)

st.title("Agent with No Memory")

task = st.text_area(f"Enter task here", key="task")
if st.button("Run", key="generate_script"):
    with st.spinner("Processing"):
        no_memory_test(task)
        st.success("Process done!")

