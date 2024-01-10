import streamlit as st
from dotenv import load_dotenv, find_dotenv
import guidance
import openai
import os
import re
from time import time as now
import sys
sys.path.append("/Users/allyne/Documents/GitHub/Unity-Agent/")

PROMPT = '''
{{#system~}}
You are a skilled C# developer tasked with crafting a C# script for a 3D ubicomp space. 
{{~/system}}
{{#user~}}
Your work revolves around our proprietary C# system. Our system comprises of: 
- SceneAPI: This is the wrapper class for our ubicomp space. The methods here will allow for manipulating the whole space. 
- Object3D: Each object in the space is of this type, and the methods here are available for every object. The anchor for the position and rotation of each object is at the bottom center of the object. 
- Vector3D: Any 3-dimensional dataset uses this class to represent x, y, and z.
- Color3D: Color information using RGBA.
                
**As the script's class inherits from `SceneAPI`, you can directly call its methods without prefixing.**

Use the provided system to manipulate the room.

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
// When performing calculations like multiplication remember to convert all your Vector3Ds to Vector3s using ToVector3() before performing the calculations.
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
Follow these steps:
1. Create a public class [classname] : SceneAPI where [classname] should be indicative of the user's task context.
2. Write method(s) using C# code to implement the task. 
3. Use Debug.Log statements for action logging and error handling.
4. dhere strictly to standard C# methods. 
5. Add comments to your code to explain your thought process.

The user task is: 
{{user_task}}

Your format for responses should strictly be: 
```
public class YourChosenClassName : SceneAPI
{	
    // Add any needed class members here
    
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

def generate_script(user_task,model_name):
    guidance.llm = guidance.llms.OpenAI(model_name)
    coder = guidance(PROMPT)
    resp = coder(user_task=user_task)
    create_and_download_cs_file(resp["script"])

def edit_code_string(code_string):
    try:
        first_index = code_string.index("public class")
        last_index = code_string.rindex("}")  
        return code_string[first_index:last_index+1]   
    except ValueError:
        st.write("Invalid code: 'using' or '}' not found.")

def get_class_name_from_code(code_string):
    # Extract the class name from the code string using regex
    match = re.search(r'public class (\w+)', code_string)
    if match:
        return match.group(1)
    return "generated_script" 

def create_and_download_cs_file(code_string):
    code_string = edit_code_string(code_string)
    code_string = "using UnityEngine;\nusing UnityEngine.Events;\nusing UnityEngine.XR.Interaction.Toolkit;\nusing System;\nusing System.Collections.Generic;\nusing Enums;\nusing UnityEngine.AI;\nusing System.Linq;\n\n" + code_string
    st.write("\n```csharp\n" + code_string + "\n\n")
    class_name = get_class_name_from_code(code_string)
    file_name = f"{class_name}.cs"
    file_path = f"baseline_generated_scripts/{file_name}"
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

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Testing baseline agents")

st.markdown("#### 1. gpt-3.5-turbo-1106")
task = st.text_area(f"Enter task here", key="task_3.5")
if st.button("Run", key="generate_script_3.5"):
    with st.spinner("Processing"):
        generate_script(task, "gpt-3.5-turbo-1106")
        st.success("Process done!")

st.markdown("#### 2. gpt-4-0613")
task = st.text_area(f"Enter task here", key="task_4")
if st.button("Run", key="generate_script_4"):
    with st.spinner("Processing"):
        generate_script(task, "gpt-4-0613")
        st.success("Process done!")