from dotenv import load_dotenv
import os
import json
import threading
import datetime
import time
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Reminders():
    '''
    Schema and instructions for each reminder provided in the comments above.
    '''
    def __init__(self, tts_converter, openai = None):
        self.reminders = []
        self.load()
        self.init_check_thread()
        self.openai = openai
        self.parser_obj = {
            "keywords": ["my reminders"],
            "prior_function": [
                {
                    "function": self.prior_fn,
                    "arg": {}
                }
            ]
        }

        self.tts = tts_converter
        self.init_parser_obj()

    def init_parser_obj(self):
        if self.openai:
            self.openai.add_object(self.parser_obj)

    def load(self):
        path = 'data/reminders.json'
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                json.dump(self.reminders, f)
        else:
            with open(path, 'r') as f:
                self.reminders = json.load(f)

    def get_reminders(self):
        return str(self.reminders)

    def delete(self, title, prompt):
        self.reminders = [reminder for reminder in self.reminders if reminder['title'].lower() != title.lower()]
        self.save_reminders()

        return {"messages": [
            {
                "role": "user", "content": prompt
            },
            {
                "role": "system", "content": f"Succesfully deleted the reminder titled {title}"
            }
        ]
        ,
            "histories": [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "assistant-action-taken", "content": f"succesfully deleted the reminder titled {title}"
                }
            ]
        }

    def new_reminder(self, title, prompt, description = None, due_date = None, due_time = None):
        reminder = {
            "description": description,
            "title": title
            }
        
        if due_time:
            reminder["due_time"] = due_time

        if due_date:
            reminder["due_date"] = due_date


        self.reminders.append(reminder)
        self.save_reminders()

        return {"messages": [
            {
                "role": "user", "content": prompt
            },
            {
                "role": "system", "content": F"Assistant completely created a file titled as {title}"
            }
        ],
            "histories": [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "assistant-action-taken", "content": f"succesfully created a reminder titled {title}"
                }
            ]
        }

    def update(self, title, prompt, description = None, due_date = None, due_time = None):
        for index, reminder in enumerate(self.reminders):
            if reminder["title"] == title:
                if description:
                    self.reminders[index]["description"] = description
                if due_date:
                    self.reminders[index]["due_date"] = due_date
                if due_time:
                    self.reminders[index]["due_time"] = due_time

            break
        
        self.save_reminders()

        return {
            "messages": [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "system", "content": f"Assistant has succesfully updated the reminder titled {title}"
                }
            ],

            "histories": [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "assistant-action-taken", "content": f"succesfully updated the reminder titled {title}"
                }
            ]
        }


    def save_reminders(self):
        path = 'data/reminders.json'
        with open(path, 'w') as f:
            json.dump(self.reminders, f)

    def init_check_thread(self):
        thread = threading.Thread(target=self.check_reminders, daemon=True)
        thread.start()

    def check_reminders(self):
        while True:
            now = datetime.datetime.now()
            for reminder in self.reminders:
                due_date = datetime.datetime.strptime(reminder['due_date'], '%m-%d-%Y')
                if now >= due_date:
                    print(f'Due Reminder Detected with title: {reminder["description"]}')
            print('checking')
            time.sleep(5)

            # Sends to next prompt of AI
    def reminder_titles(self):
        titles = [reminder["title"] for reminder in self.reminders]
        print(", ".join(titles))
        return ", ".join(titles)

    def prior_fn(self, arg):
        print("CALLED THIS")
        keywords = {
            "create": ["create", "make", "add"],
            "read": ["check", "read"],
            "update": ["update", "rewrite"],
            "delete": ["remove", "delete", "cancel"]
        }

        function_map = {
            "create": self.new_reminder,
            "read": self.get_reminders,
            "update": self.update,
            "delete": self.delete
        }

        action = ""
        break_all = False
        prompt_list = arg["prompt"].split(" ")
        for prop in keywords:
            for word in prompt_list:
                if word in keywords[prop]:
                    action = prop
                    break_all = True
                    break
            if break_all:
                break

        if action == "create":
            self.tts.summer_say("Writing that down now, sir.")
            response = self.openai_function_call(
                [
                    {
                        "role": "system", "content": f"Current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "user", "content": arg["prompt"]
                    }
                ], 
                {
                    "name": action,
                    "description": "Create a reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "title of the reminder",
                            },
                            "description": {
                                "type": "string",
                                "description": "description of the reminder"
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Due date of the reminder with format of %m-%d-%Y"
                            },
                            "due_time": {
                                "type": "string",
                                "description": "due time of the reminder with military time format e.g. 2300, 0700"
                            }
                        }
                    },
                    "required": ["title", "description", "due_time", "due_date"]
                },
                action
            )

        elif action == "read":
            response_obj = arg
            response_obj["messages"] = [
                {
                    "role": "user", "content": arg["prompt"]
                },
                {
                    "role": "system", "content": f"Obtained reminders: {self.get_reminders()}"
                }
            ]

            response_obj["histories"] = [
                {
                    "role": "user", "content": arg["prompt"]
                },
                {
                    "role": "assistant-action-taken", "content": "Succesfully accessed the reminders"
                }
            ]

            response_obj["delete_prompt"] = True
            return response_obj


        elif action == "update":
            self.tts.summer_say("Updating now, sir. One moment.")
            response = self.openai_function_call(
                [
                    {
                        "role": "system", "content": f"Current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "system", "content": f"available reminders: [{self.get_reminders()}]"
                    },
                    {
                        "role": "user", "content": arg["prompt"]
                    }
                ],
                {
                    "name": action,
                    "description": "Update a reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "title of the reminder to be updated",
                            },
                             "description": {
                                "type": "string",
                                "description": "updated description of the reminder if there is a change"
                            },
                            "due_date": {
                                "type": "string",
                                "description": "updated due date of the reminder if there is a change with format of %m-%d-%Y"
                            },
                            "due_time": {
                                "type": "string",
                                "description": "updated due time of the reminder if there is a changewith military time format e.g. 2300, 0700"
                            }
                        }
                    },
                    "required": ["title", "description", "due_date", "dute_time"]
                },
                action
            )

              
        elif action == "delete":
            response = self.openai_function_call(
                [
                    {
                        "role": "system", "content": f"available reminders: [{self.reminder_titles()}]"
                    },
                    {
                        "role": "user", "content": arg["prompt"]
                    }
                ],
                {
                    "name": action,
                    "description": "Delete a reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "title of the reminder to be deleted",
                            },
                        }
                    },
                    "required": ["title"]
                },
                action
            )

        fn_name = response["choices"][0]["message"]["function_call"]["name"]
        fn = function_map.get(fn_name)
        args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        args["prompt"] = arg["prompt"]

        fn_response = fn(**args)

        response_obj = arg

        if "messages" in response_obj:
            response_obj["messages"] += fn_response["messages"]
        else:
            response_obj["messages"] = fn_response["messages"]

        if "histories" in fn_response:
            response_obj["histories"] = fn_response["histories"]

        response_obj["delete_prompt"] = True
        return response_obj
        
    
    def openai_function_call(self, messages, function_obj, function_call_str):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
            functions = [
                function_obj
            ],
            function_call = {"name": function_call_str}
        )

        return response

