'''
This should be a lightweight class and only contain actions / functions for the following:
- List files (environment for actions, )
- Read file
- Write file (only in workspace)
- Search
- Think
- Final answer
- Code Interpreter
- Retrieval
- An OpenAI Assistant class will also be provided to take in a prompt and give an output after running through all the Actions it decides is necessary. 
- Note: Logging is automatically invoked whenever an Action is invoked, saving state of work_dir in log_dir

IMPORTANT: ALL FUNCTIONS SHOULD USE WORK_DIR AS CWD, OTHERWISE, IT'LL MESS WITH CONSISTENCY AND FUNCTIONS THAT ARE CALLED BY THE AGENT AND LEAD TO PROBLEMS
'''
# Note: using the existing actions for now

import openai
from openai import OpenAI
import functools

# # TODO: not surew here to put this
# openai.api_key = os.getenv("OPENAI_API_KEY")
# self.client = OpenAI(api_key=openai.api_key)
# client = self.client
# assistant = client.beta.assistants.create(
#     name="Research Agent",
#     instructions=self.system_prompt,
#     tools=TOOL_DESCRIPTIONS,
#     model=self.args.llm_name
# )
# thread = client.beta.threads.create()

TOOL_DESCRIPTIONS = [
    # {"type": "code_interpreter"},
    # {"type": "retrieval"},
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "understandFile",
    #     "description": "Use this to read the whole file and understand certain aspects. Provide detailed description on what to look for and what should be returned.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "file_name": {"type": "string", "description": "A valid file name with relative path to current directory if needed"},
    #             "things_to_look_for": {"type": "string", "description": "A detailed description on what to look for and what should be returned"}
    #         },
    #         "required": ["file_name", "things_to_look_for"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "appendSummaryToResearchLog",
    #     "description": "Append to the summary of the previous step to research log",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "content": {"type": "string", "description": "A string within 500 character limit"}
    #         },
    #         "required": ["content"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "inspectScriptLines",
    #     "description": "Use this to inspect a specific part of a python script precisely, or the full content of a short script.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "script_name": {"type": "string", "description": "A valid python script name with relative path to current directory if needed"},
    #             "start_line_number": {"type": "integer", "description": "A valid line number"},
    #             "end_line_number": {"type": "integer", "description": "A valid line number"}
    #         },
    #         "required": ["script_name", "start_line_number", "end_line_number"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "editScript",
    #     "description": "Use this to do a large but cohesive edit over a python script. Describe the edit instruction for AI assistance.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "script_name": {"type": "string", "description": "A valid python script name. An empty script will be created if it does not exist."},
    #             "edit_instruction": {"type": "string", "description": "A detailed step by step description on how to edit it."},
    #             "save_name": {"type": "string", "description": "A valid file name for saving the edited script"}
    #         },
    #         "required": ["script_name", "edit_instruction", "save_name"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "editScriptSegment",
    #     "description": "Use this to do a cohesive edit over a segment of a python script. Describe the edit instruction for AI assistance.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "script_name": {"type": "string", "description": "A valid python script name. An empty script will be created if it does not exist."},
    #             "start_line_number": {"type": "integer", "description": "A valid starting line number for the segment"},
    #             "end_line_number": {"type": "integer", "description": "A valid ending line number for the segment"},
    #             "edit_instruction": {"type": "string", "description": "A detailed step by step description on how to edit it."},
    #             "save_name": {"type": "string", "description": "A valid file name for saving the edited script segment"}
    #         },
    #         "required": ["script_name", "start_line_number", "end_line_number", "edit_instruction", "save_name"]
    #     }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "reflection",
            "description": "Use this to reflect on all past steps. Provide a detailed description on what to reflect on and what should be returned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "things_to_reflect_on": {
                        "type": "string", 
                        "description": "A detailed description on what to reflect on and what should be returned"
                    }
                },
                "required": ["things_to_reflect_on"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "retrievalFromResearchLog",
    #     "description": "Use this to retrieve relevant information from the research log. Provide a detailed description on what to look for and what should be returned.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "current_plan": {"type": "string", "description": "A detailed description of the current research plan and status"}
    #         },
    #         "required": ["current_plan"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "listFiles",
    #     "description": "Use this to navigate the file system.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "dir_path": {
    #                 "type": "string",
    #                 "description": "A valid relative path to a directory, such as \".\" or \"folder1/folder2\""
    #             }
    #         },
    #         "required": ["dir_path"]
    #     }
    #     }
    # },
    {
        "type": "function",
        "function": {
        "name": "readFile",
        "description": "Use this to read an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "A valid file name with relative path to current directory if needed"
                }
            },
            "required": ["file_name"]
        }
        }
    },
    {
        "type": "function",
        "function": {
        "name": "writeFile",
        "description": "Use this to write a file. If the file already exists, it will be overwritten.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "A valid file name with relative path to current directory if needed"
                },
                "content": {
                    "type": "string",
                    "description": "The content to be written to the file. Please know that the execute script function will execute from the same current working directory. Also that execute script will only output the stdout of the script, so do not use visualizations or other outputs that are not stdout."
                }
            },
            "required": ["file_name", "content"]
        }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "appendFile",
    #     "description": "Use this to append a file to a new location with a new name.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "file_name": {
    #                 "type": "string",
    #                 "description": "A valid file name with relative path to current directory if needed"
    #             },
    #             "content": {
    #                 "type": "string",
    #                 "description": "The content to be appended to the file"
    #             }
    #         },
    #         "required": ["file_name", "content"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "copyFile",
    #     "description": "Use this to copy a file to a new location with a new name.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "source": {
    #                 "type": "string",
    #                 "description": "A valid file name with relative path to current directory if needed"
    #             },
    #             "destination": {
    #                 "type": "string",
    #                 "description": "A valid file name with relative path to current directory if needed"
    #             }
    #         },
    #         "required": ["source", "destination"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "undoEditScript",
    #     "description": "Use this to undo the last edit of the python script.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "script_name": {
    #                 "type": "string",
    #                 "description": "A valid python script name with relative path to current directory if needed"
    #             }
    #         },
    #         "required": ["script_name"]
    #     }
    #     }
    # },
    {
        "type": "function",
        "function": {
        "name": "executeScript",
        "description": "Use this to execute the python script. The script must already exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "A valid python script name with relative path to current directory if needed. You can only execute scripts and files in the current directory."
                }
            },
            "required": ["script_name"]
        }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "pythonREPL",
    #     "description": "A python REPL. Use this to execute single line python commands.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "command": {
    #                 "type": "string",
    #                 "description": "A valid python command"
    #             }
    #         },
    #         "required": ["command"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "requestHelp",
    #     "description": "Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "request": {
    #                 "type": "string",
    #                 "description": "A detailed description on what to do"
    #             }
    #         },
    #         "required": ["request"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "finalAnswer",
    #     "description": "Use this to submit the final answer to the research problem.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "final_answer": {
    #                 "type": "string",
    #                 "description": "A detailed description of the final answer"
    #             }
    #         },
    #         "required": ["final_answer"]
    #     }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "webSearch",
    #     "description": "Use this to search the web for information.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "The query to search on the internet"
    #             }
    #         },
    #         "required": ["query"]
    #     }
    #     }
    # }
] # TODO: Can make more simple basic functions
# https://platform.openai.com/docs/assistants/tools