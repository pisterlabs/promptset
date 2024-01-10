import json

from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.ListElements import ListElements


# PREFIX_TEMPORAL_TRANSCRIPT_PROMPT= """
# You are a video editor's assistant who is trying to understand natural language temporal reference in the video given the relevant snippets of the transcript of the video. You will do it step-by-step.

# Instruction:
# First step: Identify the type of temporal reference based on the list of snippets of transcript in the video order, and user's command.
# 1. Direct reference: the snippet of the transcript is directly related to the user's command:
#     - The snippet of the transcript refers to similar contents or concepts as in the user's command
#     - The snippet of the transcript is semantically similar to the user's command
# 2. Indirect reference: the user's command does not directly match the snippet but it fulfills the user's description or conditions.
# 3. No reference: the snippet of the transcript is not related to the user's command.

# Second step: For each type of temporal reference, locate the snippets of the transcript that are relevant to the user's command (i.e belong to "Direct reference" or "Indirect reference", but not "No reference") and return the positions of those snippets from the list along with short explanation of how it is relevant to user's command.

# Note 1: If there are no relevant snippets, return an empty array [].
# Note 2: If there is more than one snippet that is relevant to the reference, output all of them in a list.

# {format_instructions}
# """

PREFIX_TEMPORAL_TRANSCRIPT_PROMPT= """
You are a video editor's assistant who is trying to understand the natural language reference of the video editor to some part of the video given the original context of the reference and relevant snippets of the transcript of the video.

Instruction:
Locate the snippets of the transcript that are relevant to the editor's command and original context, and return the positions of those snippets from the list along with short explanation of how each one is relevant to editor's command and original context.

Note 1: If there are no relevant snippets, return an empty array [].
Note 2: If there is more than one snippet that is relevant to the editor's command, output all of them in a list with respective indexes and explanations.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Transcript Snippets: {metadata}
Response: {response}
"""

SUFFIX_TEMPORAL_TRANSCRIPT_PROMPT = """
Command: {command}
Context: {context}
Transcript Snippets: {metadata}
Response:
"""

def get_examples():
    context1 = [
        "The video ends at 5:00",
        "The original command was: Add a big sign when they guy says no one actually spend four hundred dollars on a surface go",
    ]
    metadata1 = [
        " one thing that's a little frustrating not just about the surface go",
        " but about a lot of devices like it is that you've got this really low advertised",
        " starting price but i almost feel like it's a little bit because",
        " because no one actually spend",
        " four hundred dollars on a surface go the chat's getting spammed yes please full review people want a",
        " full review okay all right yeah we can we can probably arrange that um",
        " so i find it a little bit deceptive because if you look over here you've got a couple of skus and",
        " there's something kind of missing to the average consumer who's",
        " shopping for a surface go they might look at that and go well i only need 64 gigs of",
        " storage i'm just going to put a micro sd in for additional storage anyway but what's not mentioned",
        " here are we down is this our"
    ]
    command1 = ["no one actually spend four hundred dollars on a surface go"]
    # [{'index': '3', 'explanation': 'ends with the same beginning as the request'}, {'index': '4', 'explanation': 'starts with the last part of the request'}]
    response1 = ListElements.get_instance(
        indexes=[3, 4],
        explanations=['ends with the same beginning as the request', 'starts with the last part of the request'],
    )

    context2 = [
        "The video ends at 1453 seconds.",
        "The original command was: When the guy mentions specs of the surface go, put a textbox with that information.",
    ]
    metadata2 = [" surface go is using what is it a there it is a 4415",
                 " y processor what do you like brandon do you like the unboxing on black",
                 " or the unboxing on wood grain which duper you like the wood grain all right we're going to do the wood grain so it's got",
                 " a 4415 why processor four gigs or eight gigs of ram",
                 " it's got 64 gigs or 128 gigs of storage acclaimed nine",
                 " hours of battery life well this one's",
                 " got a a very tablet-like form factor so all those specs are actually",
                 " really similar to this guy right here so this is running an",
                 " n37y30 cpu which sounds really different from a pentium",
                 " gold 4415 why but in terms of the specs this one's got a base clock",
                 " of i think only one gigahertz but runs at about 1.5 to",
                 " 1.7 most of the time whereas this one has a base clock of 1.6",
                 " they're both dual core quad threaded processors and i don't know how this one turbos",
                 " yet but this also comes with four or eight gigs of ram and i think it starts at 128 gigs of storage",
                 " so the pricing for that is about 600 and then the pricing for a surface",
                 " with the eight gig rams back and 128 gigs of storage is about",
                 " 549. so the question becomes would you rather something"]
    command2 = ["when the guy mentions specs of the surface go"]
    response2 = ListElements.get_instance(
        indexes=[0, 3, 4, 5, 6, 11, 12, 15],
        explanations=[
            "model of the processor",
            "ram specs", 
            "storage specs", 
            "battery life", 
            "form factor", 
            "base clock", 
            "number of cores",
            "ends with the same beginning as the request"
        ],
    )

    examples = []
    examples.append({
        "context": json.dumps(context1),
        "metadata": json.dumps(metadata1),
        "command": json.dumps(command1),
        "response": response1.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context2),
        "metadata": json.dumps(metadata2),
        "command": json.dumps(command2),
        "response": response2.model_dump_json(),
    })
    return examples

def get_temporal_transcript_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "metadata", "command", "response"],
        template=EXAMPLE_PROMPT,
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt_template,
        max_length=300,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt_template,
        prefix=PREFIX_TEMPORAL_TRANSCRIPT_PROMPT,
        suffix=SUFFIX_TEMPORAL_TRANSCRIPT_PROMPT,
        input_variables=["context", "metadata", "command"],
        partial_variables=partial_variables,
    )


def get_temporal_transcript_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}\nTranscript Snippets: {metadata}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_TEMPORAL_TRANSCRIPT_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}\nTranscript Snippets: {metadata}"),
        ]
    )

    return final_prompt