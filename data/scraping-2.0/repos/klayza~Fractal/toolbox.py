import openai
import json
from datetime import datetime
from datetime import timedelta
import fractal

registeredTools = []
registeredMicroTools = []


def Tool(cls):
    cls.isTool = True
    registeredTools.append(cls)
    return cls


def MicroTool(cls):
    cls.isMicroTool = True
    registeredMicroTools.append(cls)
    return cls


@Tool
class AddNewTask:
    def __init__(self):
        self.needID = True
        self.func = self.addNewTask
        self.schema = {
            "name": "AddNewTask",
            "description": "Add a task / reminder to a user's todo / task list",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name / title of the task",
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the task",
                    },
                    "start": {
                        "type": "string",
                        "format": "date-time",
                        "description": "The start time of the task ",
                    },
                    "due": {
                        "type": "string",
                        "format": "date-time",
                        "description": "The due time of the task",
                    },
                    "status": {
                        "type": "string",
                        "description": "The status of the task",
                        "enum": ["unstarted", "in-progress", "completed"],
                    },
                    "priority": {
                        "type": "integer",
                        "description": "The priority of the task",
                    },
                    "importance": {
                        "type": "integer",
                        "description": "The importance of the task",
                    },
                    "comments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Comments related to the task",
                    },
                },
                "required": ["name", "description"],
            },
        }

    def addNewTask(self, userID, args):
        task = Task(**args)
        old = []
        prioritized = {}
        canWrite = False
        response = {"header": ""}
        old = fractal.getUserData(userID)

        new = {
            "name": task.name,
            "description": task.description,
            "status": task.status,
            "start": task.start,
            "due": task.due,
            "priority": task.priority,
            "importance": task.importance,
            "comments": task.comments,
        }

        if old.get("values") or old.get("interests"):
            # Will need to make eval simpler and focused on priorities when given multiple tasks (Was getting importance of 8 for doing dishes!)
            # prioritized = evalTask(new, old["values"], old["interests"])
            pass

        if prioritized:
            canWrite = True
            old["tasks"].append(prioritized)
            response["header"] = "Task added and prioritized"
            response = prioritized
        else:
            canWrite = True
            old["tasks"].append(new)
            response["header"] = "Task added"
            response["content"] = prioritized

        if canWrite:
            with open(f"Data/{userID}/User.json", "w", encoding="utf-8") as f:
                json.dump(old, f, indent=4)

        return response


@Tool
class SummarizeTasks:
    def __init__(self):
        self.needID = True
        self.func = self.summarizeTasks
        self.schema = {
            "name": "SummarizeTasks",
            "description": "Summarizes a user's tasks. Only use if a user cannot remember them, or asks for their a list of their todo / tasks / chores list",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Instructions on how the tasks should be organized",
                    },
                },
            },
        }

    def summarizeTasks(self, userID, prompt):
        tasks = fractal.getUserData(userID).get("tasks")
        tasks = [task for task in tasks if task["status"] != "complete"]
        return tasks


@Tool
class SendSelfie:
    def __init__(self):
        self.needID = True
        self.func = self.sendSelfie
        self.schema = {
            "name": "SendSelfie",
            "description": "Sends a selfie to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "current emotion in selfie",
                    },
                    "verb": {
                        "type": "string",
                        "description": "what you are currently doing in a single word",
                    },
                    "place": {
                        "type": "string",
                        "description": "where you are (ex. diner, bedroom, nature, etc.)",
                    },
                    "condition": {
                        "type": "string",
                        "description": "the state of your environment (ex. raining, night, day, thunder)",
                    },
                    # "nsfw": {
                    #     "type": "boolean",
                    #     "description": "Whether the content is NSFW (Not Safe For Work)",
                    # },
                },
            },
            "required": ["emotion", "verb", "place", "condition"],
        }

    def sendSelfie(self, userID, args):
        valList = list(args.values())

        if args.get("nsfw", False):
            pl = fractal.buildSDPayload(userID, valList, "decrepit")
        else:
            pl = fractal.buildSDPayload(userID, valList)

        path = fractal.getImage(pl)
        fractal.sendPhoto(path, userID)
        return "Selfie sent."


@Tool
class CompleteTask:
    def __init__(self):
        self.needID = True
        self.func = self.markTaskComplete
        self.schema = {
            "name": "CompleteTask",
            "description": "Mark a task complete. Only use when a user explicitly says they've completed something",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "The general name of the task",
                    }
                },
            },
            "required": ["task_name"],
        }

    def markTaskComplete(self, userID, generalTaskName):
        tasks = fractal.getUserData(userID).get("tasks")
        tasks = [task for task in tasks if task["status"] != "complete"]

        agent = Agent()
        agent.Load([SelectChoice])
        response = agent.Do(
            prompt=f"Select the choice that is the closest in meaning to '{generalTaskName}'",
            data=self.toEnglish(tasks),
        )
        if response["index"].isdigit:
            return self.setTaskStatus(userID, int(response.get("index")), "complete")
        else:
            return "Task not found"

    def toEnglish(self, tasks):
        taskString = ""
        i = 1
        for task in tasks:
            taskString += str(i) + " - " + task["name"] + "\n"
            i += 1
        return taskString

    def setTaskStatus(self, userID, index, status):
        data = fractal.getUserData(userID)
        data["tasks"][index - 1]["status"] = status
        fractal.setUserData(userID, data)
        return "Task set completed"


# user: hey I did that one task! ai: *detects user completed task* -> markTaskComplete(1349, that one task) -> loadAgent("")


@MicroTool
class SelectChoice:
    def __init__(self):
        self.needID = False
        self.func = self.selectChoice
        self.schema = {
            "name": "SelectChoice",
            "description": "Choose a number to select a choice",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "A number. The number which represents your choice.",
                    }
                },
            },
        }

    # Pseudo-function (Only need for paramaterized responses)
    def selectChoice(self, choice):
        return choice


class Agent:
    def __init__(self):
        self.availableTools = registeredMicroTools
        self.useAvailable = False
        self.loadedTools = []

    def Do(self, prompt, data):
        openai.api_key = fractal.OPENAI_API_KEY
        messages = []
        messages.append(
            {
                "role": "system",
                "content": f"These are your instructions, be organized and highly detailed: {prompt}",
            }
        )
        messages.append({"role": "user", "content": str(data)})
        functions = None
        response = None
        if self.loadedTools:
            toolInstances = {tool.__name__: tool() for tool in self.loadedTools}
            toolSchemas = [instance.schema for instance in toolInstances.values()]
            functions = toolSchemas
        elif self.useAvailable and self.availableTools:
            toolInstances = {tool.__name__: tool() for tool in self.availableTools}
            toolSchemas = [instance.schema for instance in toolInstances.values()]
            functions = toolSchemas
        if functions:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                functions=functions,
                function_call="auto",
            )
            responseMsg = response["choices"][0]["message"]

            if responseMsg.get("function_call"):
                chosenTool = toolInstances.get(responseMsg["function_call"]["name"])
                functionToCall = chosenTool.func

                functionJsonArgs = json.loads(responseMsg["function_call"]["arguments"])

                return functionToCall(functionJsonArgs)

        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613", messages=messages
            )

        return response
        # return response["choices"][0]["message"].get("content")

    def Load(self, toolNames):
        for name in toolNames:
            self.loadedTools.append(name)


class Task:
    start_default = datetime.today().strftime("%m-%d-%Y %H:%M")
    due_default = (datetime.today() + timedelta(days=7)).strftime("%m-%d-%Y %H:%M")

    def __init__(
        self,
        name,
        description,
        start=start_default,
        due=due_default,
        status="unstarted",
        priority=None,
        importance=None,
        comments=None,
    ):
        self.name = name
        self.description = description
        self.status = status
        self.start = start
        self.due = due
        self.priority = priority
        self.importance = importance
        self.comments = comments if comments else []


def getAvailableTools():
    return registeredTools


def genSchema(obj):
    pass


def evalTask(task, values=None, interests=None):
    """Given a list of values and interests, an agent will evaluate the
    importance and priority of a task and make changes to a task"""
    if not values or not interests:
        return None

    openai.api_key = fractal.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            # {"role": "system", "content": "Given the user's interests or values, rate the priority and importance accurately"},
            {
                "role": "user",
                "content": "Analyze the user's values and interests and re-evaluate the priority and importance based on what would resonate with the user best. DATA: "
                + str({"task": task, "values": values, "interests": interests}),
            }
        ],
        functions=[
            {
                "name": "evaluateTask",
                "description": "Given the user's interests or values, evaluate each field accurately, reflecting truth",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # "start": {
                        #     "type": "string",
                        #     "format": "date-time",
                        #     "description": "The start time of the task"
                        # },
                        # "due": {
                        #     "type": "string",
                        #     "format": "date-time",
                        #     "description": "The due time of the task"
                        # },
                        # "status": {
                        #     "type": "string",
                        #     "description": "The status of the task",
                        #     "enum": ["unstarted", "in-progress", "completed"]
                        # },
                        "priority": {
                            "type": "integer",
                            "description": "The priority of the task 0 - 10",
                        },
                        "importance": {
                            "type": "integer",
                            "description": "The importance of the task 0 - 10",
                        },
                        "comments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Comments related to the task",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "The logic and reasoning behind the priority and importance values",
                        },
                    },
                    "required": ["priority", "importance", "comments", "reasoning"],
                },
            }
        ],
        function_call={"name": "evaluateTask"},
    )

    print(response)


if __name__ == "__main__":
    # print(CompleteTask().markTaskComplete(
    #     fractal.ADMIN_ID, "Clayton made some type of ramen"))
    # CompleteTask().setTaskStatus(fractal.ADMIN_ID, 4, "complete")
    # pass  # Testing goes here
    # sendSelfie_instance = SendSelfie()
    # sendSelfie_instance.sendSelfie(
    #     fractal.ADMIN_ID,
    #     emotion="happy",
    #     verb="sitting",
    #     place="diner",
    #     condition="night",
    #     nsfw=True,
    # )
    pass