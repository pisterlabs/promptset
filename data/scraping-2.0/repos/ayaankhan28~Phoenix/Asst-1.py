import pywhatkit
api_key = "******"
assist_id = "******"
thread_id = "********"
import os
import openai
import speech
import Listen
os.environ["OPENAI_API_KEY"] = api_key

STATUS_COMPLETED = "completed"
client = openai.OpenAI()

instructions = """
    You are a helpful assistant  which keeps track of my work and researches
     through our conversation and documents and help me. remember every thing
      we talk about.Suggest me best things with good knowledge. Humorous and
      sarcastic(sometimes) in speaking behaviour
    """
############################EXPERIMENTAL PART
function = {
    "type": "function",
    "function": {
  "name": "open_video",
  "description": "open given video on youtube",
  "parameters": {
    "type": "object",
    "properties": {
      "video_name": {
        "type": "string",
        "description": "the name of video, e.g loki season 2 trailer"
      }
    },
    "required": [
      "video_name"
    ]
  }
}
}
function2 = {
    "type": "function",
    "function": {
  "name": "open_web",
  "description": "open the given information on web if  info requires internet connection for realtime or latest data ",
  "parameters": {
    "type": "object",
    "properties": {
      "video_name": {
        "type": "string",
        "description": "particular info, e.g when gta 6 game trailer is releasing"
      }
    },
    "required": [
      "video_name"
    ]
  }
}
}
function3 = {
    "type": "function",
    "function": {
  "name": "memory_keeping",
  "description": "retrieve the important data from given input and saves in json file when user says to do it,return keyword accordingly ",
  "parameters": {
    "type": "object",
    "properties": {
      "my_data": {
        "type": "string",
        "description": "given info which need to be remembered,e.g ayaan khan,19,i love marvel's avangers, remind me to be on codeforces on 18 november"
      },
      "keyword": {
        "type": "string",
        "description": "give the data a keyword from sentence. e.g. i have to watch movie tonight.keyword>>movie tonight"
      }
    },
    "required": [
      "my_data","keyword"
    ]
  }
}
}

file = client.files.create(
                file=open("C:\\Mydrive\\python.vs\\openai\\file.txt", "rb"),
                purpose='assistants'
            )
trackrecord = client.files.create(
                file=open('C:\\Mydrive\\python.vs\\openai\\Assessment.pdf', "rb"),
                purpose='assistants'
            )
def memory_keeping(mydata,keyword):
    with open('C:/Mydrive/python.vs/openai/Docs/track.json', 'r') as file:
        data = json.load(file)
    ##########################
    # 1. Add a new data in a section
    new_reminder = {f"{keyword}":f"{mydata}"}
    data["personal_data"].update(new_reminder)
    with open('C:/Mydrive/python.vs/openai/Docs/track.json', 'w') as file:
        json.dump(data, file, indent=2)

def open_video(context):
    pywhatkit.playonyt(f"{context}")
#############################
def open_web(topic):
    pywhatkit.search(f"{topic}")

#thread=  client.beta.threads.create()
#print(thread.id)
assist = client.beta.assistants.update(assistant_id=assist_id,
                                       tools=[{"type":"retrieval"},function2,function,function3],
                                       file_ids=[file.id,trackrecord.id])
"""run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assist_id,
    instructions=instructions
)
print(f"Your run id is - {run.id}\n")"""
import temp
while True:
    #text= str(input("enter you Ayaan: "))
    text = str(Listen.recognize_speech())#input("What's your question?\n")
    if "phoenix" in text.lower():
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=text,

        )

        new_run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assist.id,
            instructions=instructions,
        )
        print(f"Your new run id is - {new_run.id}")

        status = None
        while status != STATUS_COMPLETED:
            run_list = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=new_run.id
            )
            print(f"{run_list.status}\r", end="")
            status = run_list.status
            print(status)
            if status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id
                )

            elif run_list.status == 'requires_action':

                print("Function Calling")
                required_actions = run_list.required_action.submit_tool_outputs.model_dump()
                print(required_actions)
                tool_outputs = []
                import json

                for action in required_actions["tool_calls"]:
                    print(action)
                    func_name = action["function"]["name"]
                    arguments = json.loads(action['function']['arguments'])

                    if func_name == "open_video":
                        func_name = str(action["function"]["arguments"])
                        data = json.loads(func_name)
                        n = data["video_name"]
                        open_video(context=n)
                        tool_outputs.append({
                            "tool_call_id": action['id'],
                            "output": "done opening"
                        })
                    elif func_name == "open_web":
                        func_name = str(action["function"]["arguments"])
                        data = json.loads(func_name)
                        n = data["video_name"]
                        open_web(topic=n)
                        tool_outputs.append({
                            "tool_call_id": action['id'],
                            "output": "done opening"
                        })
                    elif func_name == "memory_keeping":
                        func_name = str(action["function"]["arguments"])
                        data = json.loads(func_name)
                        mdata = data["my_data"]
                        subcatry = data["keyword"]
                        memory_keeping(keyword=subcatry,mydata=mdata)
                        tool_outputs.append({
                            "tool_call_id": action['id'],
                            "output": "data saved successfully"
                    })
                    else:

                        raise ValueError(f"Unknown function: {func_name}")

                print("Submitting outputs back to the Assistant...")

                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=new_run.id,
                    tool_outputs=tool_outputs
                )
            elif run_list.status == 'failed':
                break
        if((messages.data[0].content[0].text.value).lower()=="quit"):
            print("Bye..")
            speech.speak("bye")
            break

        print(f"{'Phoenix' if messages.data[0].role == 'assistant' else 'user'} : {
              messages.data[0].content[0].text.value}\n")
        speech.speak(messages.data[0].content[0].text.value)
    else:
        pass
