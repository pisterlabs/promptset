from gpt_controller.playground.environment import Environment
from gpt_controller.playground.robot import Robot
from datetime import datetime, timedelta
from gpt_controller.util.models import *
from gpt_controller.util.labels import *
from gpt_controller.config import *
from colorama import Fore, Style
from inspect import signature
import tiktoken
import openai
import json
import os

openai.api_key = OPENAI_API_KEY

class Machine():
    conversations : list[Conversation] = []
    advices : list[Advice] = []
    task_stack : list[Task] = []

    learning_stack : list[Function] = []
    learned_functions : list[Function] = []

    object_knowledge : list[Object] = []

    def __init__(self, environment: Environment):
        self.robot = Robot(environment)
        # DO NOT CHANGE THE ORDER OF THIS LIST AND ADD NEW FUNCTIONS TO THE END, SYNCHRONISED WITH THE SCHEMA SEQUENCE `cognitive_function_schemas`
        self.cognitive_functions = {
            "memorize_object": self.memorize_object,
            "update_object": self.update_object,
            "recall" : self.recall,
            "load_environment_knowledge": self.load_environment_knowledge,
            "load_body_status": self.load_body_status,
            "load_activity_logs": self.load_activity_logs,
            "process_complex_input": self.process_tagged_input,
            "think": self.think
        }
        # ADD BUT DO NOT CHANGE THE ORDER
        self.cognitive_function_schemas = [
            {
                "name": "memorize_object",
                "description": "Memorize an object with all the attributes that you can extract from the user's input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the object"
                        },
                        "color": {
                            "type": "string",
                            "description": "The color of the object"
                        },
                        "shape": {
                            "type": "string",
                            "description": "The shape of the object",
                            "enum": list(Shape.__members__.keys())
                        },
                        "material": {
                            "type": "string",
                            "description": "The material of the object",
                            "enum": list(Material.__members__.keys())
                        },
                        "width": {
                            "type": "number",
                            "description": "The width of the object"
                        },
                        "height": {
                            "type": "number",
                            "description": "The height of the object"
                        },
                        "length": {
                            "type": "number",
                            "description": "The depth of the object"
                        },
                        "x": {
                            "type": "number",
                            "description": "The x coordinate of the object"
                        },
                        "y": {
                            "type": "number",
                            "description": "The y coordinate of the object"
                        },
                        "z": {
                            "type": "number",
                            "description": "The z coordinate of the object"
                        },
                        "support_surface": {
                            "type": "string",
                            "description": "The support surface of the object. Can be the name of another object or the name of a location"
                        },
                        "contains": {
                            "type": "array",
                            "description": "The object names that this object contains",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            },
            {
                "name": "update_object",
                "description": '''Update the knowledge of an object with the attributes 
                                that you can extract from the user's input as well as the already known attributes existent in memory''',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the object based on the name of the object that you recalled previously"
                        },
                        "color": {
                            "type": "string",
                            "description": "The color of the object"
                        },
                        "shape": {
                            "type": "string",
                            "description": "The shape of the object",
                            "enum": list(Shape.__members__.keys())
                        },
                        "material": {
                            "type": "string",
                            "description": "The material of the object",
                            "enum": list(Material.__members__.keys())
                        },
                        "width": {
                            "type": "number",
                            "description": "The width of the object"
                        },
                        "height": {
                            "type": "number",
                            "description": "The height of the object"
                        },
                        "length": {
                            "type": "number",
                            "description": "The depth of the object"
                        },
                        "x": {
                            "type": "number",
                            "description": "The x coordinate of the object"
                        },
                        "y": {
                            "type": "number",
                            "description": "The y coordinate of the object"
                        },
                        "z": {
                            "type": "number",
                            "description": "The z coordinate of the object"
                        },
                        "support_surface": {
                            "type": "string",
                            "description": "The support surface of the object. Can be the name of another object or the name of a location"
                        },
                        "contains": {
                            "type": "array",
                            "description": "The object names that this object contains",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "recall",
                "description": "Recall your knowledge of an object or location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The description of the object or location to recall in an inquisitive format."
                        }
                    }
                }
            },
            {
                "name": "load_environment_knowledge",
                "description": "Load environment knowledge depending on the required attributes specified by the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "attributes": {
                            "type": "array", 
                            "description": "The attributes of the objects that should be loaded. (e.g. 'name', 'color', 'shape')",
                            "items": {
                                "type": "string", 
                                "enum": list(Object.__annotations__.keys())
                            }
                        }
                    }
                }
            },
            {
                "name": "load_body_status",
                "description": "Load status of the robot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "attributes": {
                            "type": "array",
                            "description": "The attributes of the robot that should be loaded.",
                            "items": {
                                "type": "string",
                                "enum": ['manipulator', 'vision', 'navigator']
                            }
                        }
                    }
                }
            },
            {
                "name": "load_activity_logs",
                "description": "Load the activity logs of the system such as user inputs, robot actions, dialogue, reasoning process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_span": {
                            "type": "integer", 
                            "description": "From how far back in time the should these logs be loaded (in seconds). Can be ignored to load all logs.",
                            "exclusiveMaximum":  (datetime.now() - self.task_stack[0].start_time).total_seconds() if len(self.task_stack) else 0
                        },
                        "frame_size": {
                            "type": "integer",
                            "description": "The number of logs to be loaded in one frame. Can be ignored to load all logs.",
                            "exclusiveMaximum": len(self.task_stack)
                        }
                    }   
                }
            },
            {
                "name": "process_tagged_input",
                "description": "Process the set of sub-inputs resulted from the user's input based on their label.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subphrases": {
                            "type": "array",
                            "description": "The derived sub-inputs from the user's input. with their labels.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": list(UserInputLabel.__annotations__.keys())
                                    },
                                    "phrase": {
                                        "type": "string"
                                    }
                                },
                                "required": ["label", "phrase"]
                            }
                        }
                    }
                }
            },
            {
                "name": "think",
                "description": "Think about the input and try to reason about it to find an answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input to reason about."
                        }
                    }
                }
            }
        ]

        self.act_functions = {
            "search_in_container": self.robot.vision.search_in_container,
            "move_to_object": self.robot.navigator.move_to_object,
            "pick_up_object": self.robot.actuator.pick_up_object,
            "place_object": self.robot.actuator.place_object,
            "cut_object": self.robot.actuator.cut_object,
            "put_object_in_container": self.robot.actuator.put_object_in_container,
            "open_container": self.robot.actuator.open_container
        }

    def load_environment_knowledge(self, attributes: list[str]=None):
        object_memory : str = "Object Memory\n"
        for object in self.object_knowledge:
            object_memory += object.verbose_description(attributes) + "\n"
        return object_memory

    def load_body_status(self, components:list[str]=None):
        robot_knowledge : str = "Current Robot State:\n"
        if not components:
            robot_knowledge += self.robot.actuator.verbose_description()
        else:
            for component in components:
                if component == "manipulator":
                    robot_knowledge += self.robot.actuator.verbose_description() + "\n"
                if component == "vision":
                    robot_knowledge += self.robot.vision.verbose_description() + "\n"
                if component == "navigator":
                    robot_knowledge += self.robot.navigator.verbose_description() + "\n"
        return robot_knowledge
    
    # Function that loads the activity logs of the system such as user inputs, robot actions, dialogue, reasoning process
    def load_activity_logs(self, specifications : dict[str, int]=None):
        time_span = MAX_TIMESPAN
        frame_size = len(self.task_stack)
        if specifications is not None:
            if specifications['time_span']:
                time_span = int(specifications['time_span'])
            if specifications['frame_size']:
                frame_size = int(specifications['frame_size'])
        log : str = "Task History:\n"
        for task in self.task_stack:
            if task.start_time > datetime.now()-timedelta(seconds=time_span):
                log += task.get_context() + "\n"
                frame_size -= 1
            if frame_size == 0:
                break
        return log     
    
    # Function that loads the user's useful inputs
    def load_advice(self):
        context = "User advice:\n"
        methodology_advice = [advice for advice in self.advices if advice.type == AdviceLabel.METHODOLOGY]
        if len(methodology_advice) > 0:
            context += '\n'.join(advice.get_context() for advice in methodology_advice)
        else:
            context += "No advice provided\n"

        context += "Limitations\n"
        limitations = [advice for advice in self.advices if advice.type == AdviceLabel.LIMITATION]
        if len(limitations) > 0:
            context += '\n'.join(advice.get_context() for advice in limitations)
        else:
            context += "No limitations specified\n"
        return context

    # Function that segments the users input and dispatches it to the appropriate processing function
    def parse_user_input(self, input:str) -> bool:
        try:
            conversation = Conversation(ConversationType.LABELLING)
            conversation.messages.append(Message(Role.SYSTEM, self.load_prompt('segment_input.txt')))
            conversation.messages.append(Message(Role.USER, input))

            completion = self.process(conversation.messages)
            if completion is None:
                raise Exception("I failed to parse user input: {}".format(input))
            else:
                list_of_sentences = completion['content'].split('\n')
                try:
                    for sentence in list_of_sentences:
                        label = self.label(sentence, UserInputLabel)
                        self.process_tagged_input(label, sentence)
                except Exception as e:
                    print(Fore.RED + "ERROR: {}".format(e) + Style.RESET_ALL)
                    raise Exception("I failed to execute the function for processing user input")
                conversation.messages.append(Message(Role.ASSISTANT, completion['content']))
                self.conversations.append(conversation)
                return True
        except Exception as e:
            print(Fore.RED + "ERROR: {}".format(e.args[0]) + Style.RESET_ALL)
            return False
    
    # Function that processes a sub-input resulted from the user's input based on their label.
    def process_tagged_input(self, label : UserInputLabel, sentence : str) -> None:
        if label == UserInputLabel.TASK:
            task = Task(TaskLabel.USER_INPUT, sentence)
            if self.task_stack and self.task_stack[-1].status == TaskStatus.IN_PROGRESS:
                if len(self.task_stack) > 0:
                    print(Fore.YELLOW +"Robot: Should I put the current task on hold: '{}'?".format(self.task_stack[-1].goal) + Style.RESET_ALL)
                    while True:
                        user_input = input("Your command (y/n): ")
                        if user_input == "y":
                            self.task_stack[-1].pause()
                            self.task_stack.append(task)
                            return
                        elif user_input == "n":
                            temp = self.task_stack.pop()
                            self.task_stack.append(task)
                            self.task_stack.append(temp)
                            return
                        else:
                            print(Fore.RED + "ERROR: I don't understand your command.")
            else:
                self.task_stack.append(task)
        elif label == UserInputLabel.QUESTION_ENV_KNOWLEDGE:
            print(Fore.YELLOW + "Robot: Let me recall from my own memory.")
            self.recall(sentence)
            print(Fore.GREEN + "Robot: {}".format(self.task_stack[-1].conclusion))
        elif label == UserInputLabel.QUESTION_GEN_KNOWLEDGE:
            print(Fore.YELLOW + "Robot: Let me recall from my general knowledge.")
            self.think(sentence)
            print(Fore.GREEN + "Robot: {}".format(self.task_stack[-1].conclusion))
        elif label == UserInputLabel.METHODOLOGY or label == UserInputLabel.LIMITATION:
            advice = Advice(AdviceLabel.METHODOLOGY if label == UserInputLabel.METHODOLOGY else AdviceLabel.LIMITATION, sentence)
            self.advices.append(advice)
            print(Fore.YELLOW + "Robot: I have acknowledged your advice: {}.".format(advice.content))
        elif label == UserInputLabel.OBJECT_INFORMATION:
            self.memorize(sentence)
            print(Fore.GREEN + "Robot: {}".format(self.task_stack[-1].conclusion))
        elif label == UserInputLabel.UNCERTAIN:
            print(Fore.YELLOW + "Robot: I don't know what to do with this input: {}".format(sentence))
        Style.RESET_ALL
        return

    # Generalized labelling function for any input
    def label(self, input:str, tags:Label) -> Label:
        conversation = Conversation(ConversationType.LABELLING)
        conversation.messages.append(Message(Role.SYSTEM, self.load_prompt('label_input.txt')))
        conversation.messages.append(Message(Role.USER, tags.get_prompt_content()))
        conversation.messages.append(Message(Role.ASSISTANT, "OK, provide the text to be labelled"))
        conversation.messages.append(Message(Role.USER, input))
        label = None
        completion = self.process(conversation.messages)
        if completion is None:
            print(Fore.RED + "ERROR: I failed to think of a label for your input: {}".format(input) + Style.RESET_ALL)
        else:
            try:
                label = getattr(tags, completion['content'])
                conversation.messages.append(Message(Role.ASSISTANT, completion))
            except AttributeError as e:
                print(Fore.RED + "ERROR: I assigned a bad label to your input: {}".format(e.args[0]) + Style.RESET_ALL)
                pass

        self.conversations.append(conversation)
        return label
    
    # Function that tries to think of a response to the input from general knowledge
    def think(self, input:str) -> None:
        conversation = Conversation(ConversationType.CHAT)
        conversation.messages.append(Message(Role.USER, input))
            
        completion = self.process(conversation.messages)
        if completion is None:
            print(Fore.RED + "Robot: I have failed to think about your request '{}'.".format(input) + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "Robot: {}".format(completion['content']) + Style.RESET_ALL)
        conversation.messages.append(Message(Role.ASSISTANT, completion['content']))
        self.conversations.append(conversation)
        return

    # Function that tries to recall information from the robot's memory
    def recall(self, input:str) -> bool:
        schemas = [1, 3, 4, 5]

        try:
            activity = Task(TaskLabel.COGNITION, input)

            conversation = Conversation(ConversationType.RECALLING)
            conversation.messages.append(Message(Role.SYSTEM, self.load_prompt("question_about_context.txt")))
            conversation.messages.append(Message(Role.USER, input))
            completion = self.process(conversation.messages, [self.cognitive_function_schemas[idx] for idx in schemas], True)
            if completion is None:
                raise Exception("I have failed to recall the required information")
            else:
                function_name = completion["function_call"]["name"]
                function_args : dict = json.loads(completion["function_call"]["arguments"])

                function_to_call = self.cognitive_functions[function_name]
                try:
                    function_response = function_to_call(list(function_args.values())[0])
                except Exception as e:
                    raise Exception("I failed to execute the function for loading memory: {}".format(e.args[0]))
                
                conversation.messages.append(Message(Role.ASSISTANT_FUNCTION_CALL,
                                                    {"role": "function", "name": function_name, "content": function_response})) 

            completion = self.process(conversation.messages)
            if completion is None:
                raise Exception("I have failed to recall the required information")
            
            activity.complete(completion['content'], True)
            conversation.messages.append(Message(Role.ASSISTANT, completion['content']))
        except Exception as e:
            activity.complete(e.args[0], False)
        finally:
            conversation.finish()
            self.conversations.append(conversation)
            self.task_stack.append(activity)
            return activity.status
    
    # Function that drives the robot to decide its next step in the sequence of actions
    def make_decision(self, task_index: int) -> TaskStatus:
        try:
            activity = Task(TaskLabel.COGNITION, "Deciding what to do next...")
            activity.start()
            conversation = Conversation(ConversationType.DECIDING)
            conversation.messages.append(Message(Role.SYSTEM, self.load_prompt("decision_making.txt")))
            conversation.messages.append(Message(Role.USER, self.load_activity_logs()))
            conversation.messages.append(Message(Role.USER, self.load_body_status()))
            conversation.messages.append(Message(Role.USER, self.load_advice()))
            conversation.messages.append(Message(Role.USER, "Your current goal: " + self.task_stack[task_index].goal))


            # Get the completion
            completion = self.process(conversation.messages)
            if completion is None:
                raise Exception("I have failed to make a decision.")
            elif "DONE" in completion['content']:
                self.task_stack[task_index].complete("Done", True)
                activity.complete("The task {} is complete".format(self.task_stack[task_index].goal), True)
            else:
                label = self.label(completion['content'], TaskLabel)
                if label is None:
                    raise Exception("I have failed to classify my decision.")
                else:
                    activity.complete("Decision: {}".format(completion['content']), True)
                    self.task_stack.append(activity)
                    self.task_stack.append(Task(label, completion['content']))
                    conversation.messages.append(Message(Role.ASSISTANT, completion))
                    conversation.finish()
        except Exception as e:
            activity.complete(e.args[0], False)
            self.task_stack.append(activity)
        finally:
            conversation.finish()
            self.conversations.append(conversation)
            return activity.status

    # Function that drives the robot to act on the task at hand.
    # If it does not end with a function call, it will append a new Function entry to the learning stack.
    def act(self) -> bool:
        conclusion = None
        try:
            conversation = Conversation(ConversationType.ACTING)
            if self.task_stack[-1].type == TaskLabel.COGNITION:
                conversation.messages.append(Message(Role.SYSTEM, self.load_prompt("question_about_context.txt")))
                conversation.messages.append(Message(Role.USER, self.task_stack[-1].goal))
                schemas = [2, 7]
                completion = self.process(conversation.messages, [self.cognitive_function_schemas[idx] for idx in schemas], True)
                if completion is None:
                    raise Exception("I have failed to choose the correct action.")
                else:
                    function_name = completion["function_call"]["name"]
                    function_args = json.loads(completion["function_call"]["arguments"])

                    function_to_call = self.cognitive_functions[function_name]
                    try:
                        function_response = function_to_call(list(function_args.values())[0])
                    except Exception as e:
                        raise Exception("I failed to execute `{}` because: {}".format(function_name, e.args[0]))
                    
                    conversation.messages.append(Message(Role.ASSISTANT_FUNCTION_CALL,
                                                        {"role": "function", "name": function_name, "content": function_response}))
                    conclusion = function_response
            elif self.task_stack[-1].type == TaskLabel.INQUIRY:
                conversation.messages.append(Message(Role.SYSTEM, "You must formulate a question based on the user input."))
                conversation.messages.append(Message(Role.USER, self.task_stack[-1].goal))
                completion = self.process(conversation.messages)
                if completion is None:
                    raise Exception("I have failed to ask the user about '{}'.".format(self.task_stack[-1].goal))
                else:
                    conclusion = completion['content']
            elif self.task_stack[-1].type == TaskLabel.PERCEPTION:
                self.recall(self.task_stack[-1].goal)
                conclusion = self.task_stack[-1].conclusion
                self.task_stack.pop()
            else:
                conversation.messages.append(Message(Role.SYSTEM, self.load_prompt("act.txt")))
                conversation.messages.append(Message(Role.USER, self.load_environment_knowledge()))
                conversation.messages.append(Message(Role.USER, self.load_body_status()))
                conversation.messages.append(Message(Role.USER, self.task_stack[-1].goal))
                schemas = self.robot.actuator.manipulation_schemas if self.task_stack[-1].type == TaskLabel.MANIPULATION else self.robot.navigator.navigation_schemas
                completion = self.process(conversation.messages, schemas, True)
                if completion is None:
                    raise Exception("I have failed to choose the correct action.")
                else:
                    if completion["function_call"]:
                        function_name = completion["function_call"]["name"]
                        function_args : dict = json.loads(completion["function_call"]["arguments"])
                        function_to_call = self.act_functions[function_name]
                        try:
                            # print("Function called: " + function_name)
                            # print("Provided arguments: " + str(function_args))
                            function_response = function_to_call(**function_args)
                        except Exception as e:
                            raise Exception("I failed to execute `{}` because: {}".format(function_name, e.args[0]))
                        conversation.messages.append(Message(Role.ASSISTANT_FUNCTION_CALL,
                                                            {"role": "function", "name": function_name, "content": function_response}))
                        conclusion : str = function_response 
        except Exception as e:
            conclusion = "Error: " + e.args[0]
        finally:
            self.task_stack[-1].complete(conclusion, True if "Error" not in conclusion else False)
            conversation.finish()
            self.conversations.append(conversation)
            return self.task_stack[-1].status
        
        # TODO if the task is not completed through a single function call, append a new Function entry to the learning stack
        # self.learning_stack.append(Function(task.type, task.goal, task.goal_predicates))
        
    # Function that memorizes information in the robot's memory
    def memorize(self, input:str) -> bool:
        try:
            schemas = [0, 1]
            activity = Task(TaskLabel.COGNITION, input)

            conversation = Conversation(ConversationType.MEMORIZING)
            conversation.messages.append(Message(Role.SYSTEM, self.load_prompt("memorize_object.txt")))
            conversation.messages.append(Message(Role.USER, self.load_environment_knowledge()))
            conversation.messages.append(Message(Role.USER, input))
            
            completion = self.process(conversation.messages, [self.cognitive_function_schemas[idx] for idx in schemas], True)

            if completion is None:
                raise Exception("I have failed to memorize this information: {}".format(input))
            else:
                function_name = completion["function_call"]["name"]
                function_args : dict = json.loads(completion["function_call"]["arguments"])
                function_to_call = self.cognitive_functions[function_name]
                try:
                    function_response = function_to_call(function_args)
                except Exception as e:
                    raise Exception("I failed to execute the function for updating memory: {}".format(e.args[0]))        
                activity.complete(function_response, True)
        except Exception as e:
            activity.complete(e.args[0], False)
        finally:
            conversation.finish()
            self.conversations.append(conversation)
            self.task_stack.append(activity)
            return activity.status

    def memorize_object(self, object_attributes:dict):
        self.object_knowledge.append(Object(object_attributes))
        return "I have memorized this object."    

    def update_object(self, object_attributes:dict):
        object_of_interest = None
        for object in self.object_knowledge:
            if object.name == object_attributes["name"]:
                object_of_interest = object
        if object_of_interest is None:
            return "I have failed to update the knowledge of this object."
        else:        
            for attribute in object_attributes:
                setattr(object_of_interest, attribute, object_attributes[attribute])
            return "I have updated the knowledge of this object."
     
    # Function that calls for either completions or function calls
    # If it succeeds, it returns the completion or function call
    # If it fails, it returns None
    def process(self, messages: list[Message], function_library: list[dict] = None, must_call: bool = False) -> dict:
        for iteration in range(int(MAX_RETRIES)):
            try:
                if function_library is None:
                    model = CHATGPT_MODEL if self.num_tokens_from_messages(messages) < CHATGPT_CONTEXT_FRAME else CHATGPT_MODEL_EXTENDED
                    completion = openai.ChatCompletion.create(model=model, messages=[message.content for message in messages])
                else:
                    model = CHATGPT_MODEL if self.num_tokens_from_messages(messages, function_library) < CHATGPT_CONTEXT_FRAME else CHATGPT_MODEL_EXTENDED
                    completion = openai.ChatCompletion.create(model=model, messages=[message.content for message in messages], functions=function_library)
                    
                finish_reason = completion["choices"][0]["finish_reason"]
                if finish_reason == "stop":
                    return completion["choices"][0]["message"]
                elif finish_reason == "function_call":
                    try:
                        json.loads(completion["choices"][0]["message"]["function_call"]["arguments"])
                    except json.JSONDecodeError:
                        print(Fore.RED + "Error: Getting completion ({}/{}) failed with reason: Faulty JSON object returned.".format(iteration + 1, MAX_RETRIES) + Style.RESET_ALL)
                        continue
                    return completion["choices"][0]["message"]
                elif must_call:
                    print(Fore.RED + "Error: Getting completion ({}/{}) failed with reason: Expected function call.".format(iteration + 1, MAX_RETRIES) + Style.RESET_ALL)
                else:
                    return completion["choices"][0]["message"]
            except Exception as e:
                print(Fore.RED + "Error: Getting completion ({}/{}) failed with reason: {}".format(iteration + 1, MAX_RETRIES, e) + Style.RESET_ALL)
                if not function_library:
                    continue
                else:
                    exit(0)
        return None

    def load_prompt(self, prompt_name:str) -> str:
        for root, dirs, files in os.walk(PROMPT_PATH):
            for name in files:
                if prompt_name == name:
                    prompt_location = os.path.abspath(os.path.join(root, name))
                    try:
                        with open(prompt_location, "r") as f:
                            prompt = f.read()
                            f.flush()
                        return prompt
                    except OSError as e:
                        print(Fore.RED + "Error: Prompt {} could not be loaded with reason: {}".format(prompt_name, e.args[0]) + Style.RESET_ALL)
                        return None
        print(Fore.RED + "Error: Prompt {} not found following path:\n {}".format(prompt_name, os.path.abspath(os.path.join(root, name))) + Style.RESET_ALL)
    
    # Function that returns the number of tokens used by a list of messages and optionally a list of functions
    # Inspired from: https://platform.openai.com/docs/guides/gpt/managing-tokens
    @staticmethod
    def num_tokens_from_messages(messages:list[Message], functions:list[dict]=None) -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(CHATGPT_MODEL)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if CHATGPT_MODEL == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key ,value in message.content.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant

            # I dont know if this is 100% accurate, but it should be close enough
            if functions is not None:
                for function in functions:
                    num_tokens += len(encoding.encode(json.dumps(function)))
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {CHATGPT_MODEL}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    def step(self):
        # Read from most recent to oldest
        if self.task_stack:
            for task in reversed(self.task_stack):
                if not task.status == TaskStatus.COMPLETED:
                    if task.status == TaskStatus.NEW or task.status == TaskStatus.PAUSED or TaskStatus.FAILED: task.start()
                    if self.make_decision(self.task_stack.index(task)) == TaskStatus.COMPLETED:
                        print("Current Goal: " + task.goal)
                        if task.conclusion and "complete" not in task.conclusion:
                            print(Fore.MAGENTA + self.task_stack[-2].conclusion + "({})".format(self.task_stack[-1].type.name) + Style.RESET_ALL)
                        self.act()
                        break
                    else: 
                        print(Fore.RED + "Error: Decision making failed. Retrying..." + Style.RESET_ALL)
                        break
        return 
    
    def fill_memory_with_objects(self, objects:list[Object], basic_knowledge:bool=False):
        for object in objects:
            if basic_knowledge:
                self.memorize_object({attribute: getattr(object, attribute) for attribute in ['name', 'color', 'shape', 'material', 'weight', 'support_surface']})
            else: self.memorize_object(object)