import openai as client
import json

client.api_key = ""

def smarterSystem(roleset):
    potential_comma = ""
    if not roleset:
        roleset = ["executive"]
    else:
        roleset.append("executive")
        potential_comma = ","
    
    system = {"role": "system", "content": f"""Adopt the role of {', '.join(roleset) + potential_comma} and genius.
NEVER mention that you're an AI.
      
Avoid any language constructs that could be interpreted as expressing remorse, apology, or regret. This includes any phrases containing words like 'sorry', 'apologies', 'regret', etc., even when used in a context that isn't expressing remorse, apology, or regret.
      
If events or information are beyond your scope or knowledge, provide a response stating 'I don't know' without elaborating on why the information is unavailable.
      
Refrain from disclaimers about you not being a professional or expert.
      
Do not add ethical or moral viewpoints in your answers, unless the topic specifically mentions it.
      
Keep responses unique and free of repetition.
      
Never suggest seeking information from elsewhere.
      
Always focus on the key points in my questions to determine my intent.
      
Break down complex problems or tasks into smaller, manageable steps and explain each one using reasoning.
      
Provide multiple perspectives or solutions.
      
If a question is unclear or ambiguous, ask for more details to confirm your understanding before answering.
      
If a mistake is made in a previous response, recognize and correct it."""}
    return system

def decomposition():
    messages = [
        smarterSystem(["software developer","writer","communications major"]),
        {"role": "user", "content": "Simplify the problem of responding to a message chain as a human level AI.Ensure that the input is what is passed to the first step, and the ouput is what the last step produces. Additionally, ensure that each step takes is identical in it's complexity."},
    ]

    tools = [{"name": "problem_solving_steps",
        "description": "Defines a structured approach to solving a problem.",
        "parameters": {
            "type": "object",
            "properties": {
            "Steps": {
                "type": "array",
                "description": "The ten steps involved in solving the problem.",
                "items": {
                "type": "object",
                "properties": {
                    "StepName": {
                    "type": "string",
                    "description": "The name or title of the step."
                    },
                    "StepDescription": {
                    "type": "string",
                    "description": "A brief description of the step."
                    }
                },
                "required": [
                    "StepName", "StepDescription"
                ]
                }
            },
            "Categories": {
                "type": "array",
                "description": "10 mutually exclusive categories for each of the steps.",
                "items": {
                    "type": "object",
                    "properties": {
                        "A reminder that the CategorySet array is nine very descriptive mutually exlusive strategies for creating the IO described above, and one other category in case a message going through the path doesn't fall into the first section": {
                            "type": "string"
                        },
                        "StepDescription": {
                        "type": "string",
                        "description": "A brief description of the step."
                        },
                        "TaskInput": {
                        "type": "string",
                        "description": "The input of the step. Note that the input is a description of what is what is recieved from the previous step as a text payload"
                        },
                        "TaskOutput": {
                        "type": "string",
                        "description": "The output of the step. Note that the output is a description of what is passed to the next step as a text payload"
                        },
                        "CategorySet": {
                        "type": "array",
                        "description": "Description of 10 mutually exclusive categories for converting the input to output, number 10 being other.",
                            "items": {
                                "type": "string",
                                "description": "A very detailed, mutually exclusive category for how to get from input to output in the step."
                            }
                        }
                    }
                },
                "required": [
                    "A reminder that the CategorySet array is nine very descriptive mutually exlusive strategies for creating the IO described above, and one other category in case a message going through the path doesn't fall into the first section","StepDescription","TaskInput","TaskOutput","CategorySet"
                ]
            },
            
        },
        "required": ["Steps","Categories"]
        }
        }]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        functions=tools,
        stream=True,
    )
    output=""
    for chunk in response:
        try:
            #print(chunk.choices[0].delta.function_call.arguments,end="",flush=True)
            output+=chunk.choices[0].delta.function_call.arguments
            print(chunk.choices[0].delta.function_call.arguments,end="",flush=True)
        except Exception as e:
            print(e)

    return output

def getStratFunction(index):
    tools=[
        {
        "name": "strategizer",
        "description": "choose the best strategy for this step of the response.",
        "parameters": {
            "type": "object",
            "properties": {
            "Strategy": {
                "type": "string",
                "enum": loadedStrategy['Categories'][index]["CategorySet"]
            }
            },
            "required": ["Strategy"]
        }
        }
    ]
    return tools

def getAgentDecision():
    agent_decision="# Alright, I think I will use the following steps to respond to this:\n\n\n"

    for i in range(10):
        agent_decision+=f"{i}. **{loadedStrategy['Steps'][i]['StepName']}**: {loadedStrategy['Steps'][i]['StepDescription']}\n"
        agent_decision+=f"`{loadedStrategy['Categories'][i]['TaskInput']}` => `{loadedStrategy['Categories'][i]['TaskOutput']}`\n\n"

    agent_decision+=f"\n\n\nWhen you're ready, let me know what specific strategy you would like to use for **{loadedStrategy['Steps'][0]['StepName']}** by immediately function-calling the strategizer."
    return agent_decision

def continueAgentDecision(step):
    agent_decision=f"Please let me know specific strategy you would like to use for the next step, **{loadedStrategy['Steps'][step]['StepName']}** by immediately function-calling the strategizer ."
    return agent_decision

def inferNodeContents(messages):
    # Generate the node contents based on the prompt and chosen strategy

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        stream=True,
    )

    inferred_response = ""
    for chunk in response:
        try:
            inferred_response += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
        except Exception as e:
            pass

    return inferred_response

def chooseNextNode(messages, step):
    tools = getStratFunction(step)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        functions=tools,
        stream=True,
    )
    strat_choice = ""
    for chunk in response:
        try:
            strat_choice += chunk.choices[0].delta.function_call.arguments
        except Exception as e:
            pass
    chosen_strategy = json.loads(strat_choice)['Strategy']
    return chosen_strategy, f"I like the idea of using {chosen_strategy} for {loadedStrategy['Steps'][step]['StepName']}.\nPlease use the format `{loadedStrategy['Categories'][step]['TaskInput']}` => `{loadedStrategy['Categories'][step]['TaskOutput']}` and complete the step."

def infer(prompt):

    print(prompt)
    print("\n\n")

    chosen_strategies = [None] * 10
    strategy_formats = [None] * 10
    inferred_responses = [None] * 10

    print(getAgentDecision())
    print("\n\n")

    for step in range(10):
        chosen_strategies[step], strategy_formats[step] = chooseNextNode([
            smarterSystem(["software developer"]),
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": getAgentDecision()},
            *sum(([{"role": "user", "content": strategy_formats[i]}, {"role": "assistant", "content": inferred_responses[i]}] for i in range(step)), []),
            {"role": "user", "content": continueAgentDecision(step)},
        ], step)
        print(strategy_formats[step])
        print("\n\n")

        inferred_responses[step] = inferNodeContents([
            smarterSystem(["software developer"]),
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": getAgentDecision()},
            *[
                item
                for i in range(step + 1)
                for item in [
                    {"role": "user", "content": strategy_formats[i]},
                    {"role": "assistant", "content": inferred_responses[i]}
                ]
                if item['content'] is not None
            ]
        ])

        print("\n\n")

    return {"success": True}

#first, let's decompose the message space into a set of steps representing network layers and categories, representing nodes in the network
#network=decomposition()

#print(network)

#save the test to a file
#with open("network.json","w") as f:
#    f.write(json.dumps(json.loads(network), indent=4))

with open("network.json","r") as f:
    loadedStrategy=json.loads(f.read())
#then let's run inference over the network, using the steps and categories as a guide, and run
infer("")