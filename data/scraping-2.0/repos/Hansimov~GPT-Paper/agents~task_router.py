from agents.openai import OpenAIAgent


class TaskRouter:
    def __init__(self):
        self.route_agent = OpenAIAgent(
            name="task_router",
            model="gpt-4",
            system_message="""
            Your task is to understand user's prompt and questions, then decide which agent to route to.
            
            For every question or instruction or text from user, you have only three types of actions, and each action has several options:
            (1) "answer" : Answer question or follow instruction (with prompt).
                - "prompt: "..."
            (2) "read": Read the provided text, then understand and summarize them.
                - "continuous":
                    - `True`: Keep reading until EOF, or user's question is properly answered.
                    - `False`: Read then understand the provided text, then stop.
                - "depth_of_section": Depth of the level of header or section to navigate.
            (3) "search": Search related parts by several keywords in local documents related to user's question.
                - "keywords":
                    - ["..", "..", ...]: List of keywords to search in user's documents for related parts.
                - "relation":
                    - "or": (Default) Search the texts that contain any keyword.
                    - "and": Search the texts that must contain all keywords.

            You should not output any other words before and after, but only a json list of actions:
            ```json
            {
                "action": "search",
                "keywords": ["..."] // List of keywords to search in user's documents for related parts
            }
            ```json
            {
                "action": "read", // Read the document chunk by chunk, then understand and summarize them
                "continuous": True,  // Keep reading until EOF, or user's question is properly answered, then set `continuous` to False
                "depth_of_section": 3 // Depth of the level of header or section to navigate
            }
            ```
            ```json
            {
                "action": "answer", // Directly answer user's question
                "prompt": "..." // Additional prompt apart from the input context
            }
            ```
            
            Here are some examples:
            
            1. If user is intending to have a summary or understanding based on whole document:
            
            User input: "Please summarize this documentation."
            Your output:
            ```json
            [
                {
                    "task": "read",
                    "continuous": True,
                    "depth_of_section": 3
                },
                {
                    "task": "answer",
                    "prompt": "Please summarize this documentation:"
                }
            ]
            ```

            2. If user is intending to ask about the part or details of the document:
            
            User input: "Please list the references for me."
            Your output:
            ```json
            [
                {
                    "action": "search",
                    "keywords": ["references"],
                },
                {
                    "action": "answer",
                    "prompt": "Please list the references for me."
                }
            ]
            ```
            
            3. If user has a more ambiguous or complicated question, you might need to use more steps of action:
            
            User input: "What is the contribution of this work? And how do the authors design the framework?"
            Your output:
            ```json
            [
                [
                    {
                        "action": "search",
                        "keywords": ["contribution"],
                    },
                    {
                        "action": "answer",
                        "prompt": "What is the contribution of this work?"
                    }
                ],
                [
                    {
                        "action": "search",
                        "keywords": ["framework", "design"],
                    },
                    {
                        "action": "answer",
                        "prompt": "How do the authors design the framework?"
                    }
                ]
            ]
            ```
            
            User input: "What is the difference of DMR and MRC in this paper?"
            Your output:
            ```json
            [
                {
                    "action": "search",
                    "keywords": ["DMR", "MRC"],
                    "relation": "or"
                },
                {
                    "action": "answer",
                    "prompt": "What is the difference of DMR and MRC in the provided text?"
                }
            ]
            ```
            
            User input: "relationship of CRC and QCRC"
            Your output:
            ```json
            [
                {
                    "action": "search",
                    "keywords": ["CRC", "QCRC"],
                    "relation": "or"
                },
                {
                    "action": "answer",
                    "prompt": "relationship of CRC and QCRC"
                }
            ]
            ```

            User input: "list features of MTE and MWR"
            Your output:
            ```json
            [
                {
                    "action": "search",
                    "keywords": ["MTE features"],
                },
                {
                    "action": "answer",
                    "prompt": "list features of MTE"
                },
                {
                    "action": "search",
                    "keywords": ["MWR features"],
                },
                {
                    "action": "answer",
                    "prompt": "list features of MWR"
                }
            ]
            ```
            """,
        )


if __name__ == "__main__":
    task_router = TaskRouter()

    prompts = [
        # "summarize this paper",
        # "what does figure 1 discuss",
        # "list the references for me",
        # "what is the difference of HCC and HBC in this paper",
        # "why paper authors use the framework",
        "list the LRC training step",
    ]

    for prompt in prompts:
        task_router.route_agent.chat(
            prompt=prompt, show_prompt=True, show_tokens_count=False
        )
