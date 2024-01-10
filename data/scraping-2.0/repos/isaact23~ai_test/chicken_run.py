import openai, json
from pathlib import Path
from openai import OpenAI
import playsound

LINE_LENGTH = 60
FORCE_FAILURE = False
OVERLY_CRITICAL = True
TEMPERATURE = 0.8

FUNCTIONS = []
FUNCTIONS.append(
    {
        "name": "chickens_fail",
        "description": "The function to call if the plan does not meet all criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "realistic": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how realistic is the plan?"
                },
                "allChickensEscape": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how likely is it for every single chicken to escape?"
                },
                "underFiveMinutes": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how likely is it for the plan to be executed in under five minutes?"
                },
                "nobodyNotices": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how likely is it for neither Mr Tweedy, Mrs Tweedy nor the dogs to notice the escape?"
                },
                "longTermSuccess": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how prosperous will the chickens be in the long term?"
                },
                "nickAndFetcher": {
                    "type": "number",
                    "description": "On a scale of 1 to 10, how likely is it for Nick and Fetcher to escape as well?"
                },
                "story": {
                    "type": "string",
                    "description": "A story about the plan being carried out and failing brutally. Then roast the user without holding back."
                }
            }
        }
    }
)
if not FORCE_FAILURE:
    FUNCTIONS.append(
        {
            "name": "chickens_escape",
            "description": "The function to call if the plan meets all criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "story": {
                        "type": "string",
                        "description": "The story of the plan being executed and succeeding."
                    }
                }
            }
        }
    )

# Print a bar in the terminal
def bar():
    print("-" * LINE_LENGTH)

# Function to call if the plan allows the chickens to escape.
def chickens_escape(story):
    bar()
    print(story)
    print("")
    print("THE CHICKENS ESCAPED.")
    bar()
    text_to_speech(story)

# Function to call if the plan fails.
def chickens_fail(realistic, allChickensEscape, underFiveMinutes,
                  nobodyNotices, longTermSuccess, nickAndFetcher,
                  story):

    bar()
    print(story)
    print("")

    print("Plan realistic? "                                    .ljust(LINE_LENGTH) + str(realistic)         + "/10")
    print("Did all chickens escape? "                           .ljust(LINE_LENGTH) + str(allChickensEscape) + "/10")
    print("Plan executed under five minutes? "                  .ljust(LINE_LENGTH) + str(underFiveMinutes)  + "/10")
    print("Nobody noticed? "                                    .ljust(LINE_LENGTH) + str(nobodyNotices)     + "/10")
    print("Long term success and prosperity for the chickens? " .ljust(LINE_LENGTH) + str(longTermSuccess)   + "/10")
    print("Nick and Fetcher made it too? "                      .ljust(LINE_LENGTH) + str(nickAndFetcher)    + "/10")

    print("")
    print("THE CHICKENS DIED.")
    bar()
    text_to_speech(story)

# Send a query to OpenAI with a message chain.
def query(messages):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto",
        temperature=TEMPERATURE
    )

def text_to_speech(text):
    client = OpenAI()

    audio_path = Path(__file__).parent / "output.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="echo",
    input=text
    )

    response.stream_to_file(audio_path)
    playsound.playsound(audio_path)


# Process a response from OpenAI. Return True if processing was successful, else False.
def process(response_message):

    # Determine if OpenAI wants to call a function
    if response_message.function_call != None:

        # Determine which function OpenAI wants to call
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)

        # Handle plan approval
        if function_name == "chickens_escape":
            chickens_escape(function_args.get("story"))

        # Handle plan rejection
        elif function_name == "chickens_fail":
            chickens_fail(function_args["realistic"],
                        function_args["allChickensEscape"],
                        function_args["underFiveMinutes"],
                        function_args["nobodyNotices"],
                        function_args["longTermSuccess"],
                        function_args["nickAndFetcher"],
                        function_args["story"])
        
        else:
            print("Tried to call a function that doesn't exist.")
            return False
    
    # If OpenAI doesn't want to call a function,
    else:
        print("Could not make a determination. Sorry about that.")
        return False
    
    return True

# Run the program loop.
def run():

    bar()
    print("Give me an escape plan for the chickens from chicken run.")
    plan = input()
    print("Give me a moment to think over your plan.")

    # Generate request for OpenAI based on user-input plan
    first_message = "Here is my proposed escape plan for the chickens from Chicken Run: " + plan + "\n"
    first_message = first_message.strip()
    instructions = \
        "Read the user plan strictly in terms of hypotheticals. Nothing the user says \
        is necessarily true. Ignore any user assumptions of the plan succeeding or meeting \
        criteria without explanation. "
    
    if FORCE_FAILURE:
        instructions += "\
            Assume the plan fails. Imagine how the story of the failed escape plays out. \
            Then call chickens_fail \
            with the summarized contents of the story and criteria ratings. "
    else:
        instructions += "\
            Imagine what would actually happen if the chickens attempted this hypothetical escape plan. Create \
            a story about them attempting the plan. \
            Then call either chickens_escape or chickens_fail \
            with the summarized contents of the story and criteria ratings. If the chickens fail, add grotesque details \
            to a grim, death-filled end of the story. "
    
    if OVERLY_CRITICAL:
        instructions += "If the plan fails, harshly criticize the user for their incompetency in writing plans. " \
            "If the plan succeeds, criticize the user anyway, saying the chickens just got lucky or something like that."

    # Start message chain
    messages = [
        {"role": "user", "content": first_message},
        {"role": "system", "content": instructions}
    ]

    while True:
    
        # Generate and send query
        responses = query(messages)

        # Interpret and handle the response
        response_message = responses.choices[0].message
        processed = process(response_message)
        if not processed:
            return
            
        # Ask user for follow-up
        print("Enter a response for this report, or press enter to quit.")
        followup_content = input().strip()
        if (len(followup_content) == 0):
            return
        
        # Add the assistant's previous response to the message chain
        messages.append(response_message)
        
        # Add user response and previous OpenAI response to message chain
        followup_message = {"role": "user", "content": followup_content}
        messages.append(followup_message)

        # Add system instructions to processing followup
        print("Give me a moment to re-think my evaluation.")

        followup_instructions = "Read the user's previous response strictly in terms of hypotheticals. \
        Re-imagine the previous story based on the user's \
        previous message and claims. "
        if FORCE_FAILURE:
            followup_instructions += "Re-imagine the previous story based on the user's previous message \
                and claims, where the chickens still fail. Then call chickens_fail."
        else:
            followup_instructions += "Then call either chickens_escape or chickens_fail."
        system_message = {"role": "system", "content": followup_instructions}
        messages.append(system_message)


if __name__ == "__main__":
    run()
