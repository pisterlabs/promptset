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


PREFIX_TEMPORAL_VISUAL_PROMPT= """
You are a video editor's assistant who is trying to understand the natural language reference of the video editor to some part of the video given the set of most relevant visual descriptions of 10-second clips of the video and original context of the command. Visual description of a 10-second clip consists of an action label which is a main action happening, an abstract caption which is an abstract description of the clip, and the dense captions, which are list of descriptions of objects that are present. Try taking into account each of them. 

Instruction:
Locate the visual descriptions that are relevant to the editor's command and original context of the command, and return the positions of those descriptions from the list along with short explanation of how each is relevant to editor's command.

Note 1: If there are no relevant viusal descriptions, return an empty array [].
Note 2: If there is more than one description that is relevant to the editor's command and original context, output all of them in a list.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Visual descriptions of 10-second clips: {metadata}
Response: {response}
"""

SUFFIX_TEMPORAL_VISUAL_PROMPT = """
Command: {command}
Context: {context}
Visual descriptions of 10-second clips: {metadata}
Response:
"""

def get_examples():

    context1 = [
        "The video is 34 minutes and 23 seconds long",
        "The original command was: Highlight around his hand when the guy is holding the laptop",
    ]
    metadata1 = [
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop with headphones on.",
            "dense_caption": "small black laptop on wooden desk,a man sitting at a desk,the opened laptop,black laptop on desk,a black and silver watch,a white wall,blue cord on the desk,a closed laptop computer,a laptop screen is turned on,the screen is white,a man with short hair,a mans right hand,white box on desk,black keyboard on laptop,the hand of a man,the man is wearing a black shirt,a laptop on a table,a laptop on a table,a white piece of paper\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man holding a laptop in his hands.",
            "dense_caption": "the opened laptop on the table,this is a laptop,a black box with wires plugged into it,a person is sitting on a wooden table,a black wrist band,a hand holding a laptop,laptop screen is on,the screen is white,laptop computer on desk,the keyboard of a laptop,the hand of a person,the brown and black flip phone\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
        {
            "action": "unboxing",
            "abstract_caption": "a man holding a laptop in his hands.",
            "dense_caption": "a man playing the wii,a black framed projector screen,a silver laptop,a person is holding a wii controller,a white apple computer,black cords on the desk,a man with brown hair\n",
        }
    ]
    command1 = ["when the guy is holding the laptop"]
    response1 = ListElements.get_instance(
        indexes=[1, 2, 4],
        explanations=["actions matched: laptop holding",
            "actions matched: holding a laptop", 
            "matching actions: holding a laptop",
        ],
    )

    context2 = [
        "The video ends at 4:34",
        "The original command was: Put a warning sign when the guy is focusing on something else than camera",
    ]
    metadata2 = [
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop with headphones on.",
            "dense_caption": "small black laptop on wooden desk,a man sitting at a desk,the opened laptop,black laptop on desk,a black and silver watch,a white wall,blue cord on the desk,a closed laptop computer,a laptop screen is turned on,the screen is white,a man with short hair,a mans right hand,white box on desk,black keyboard on laptop,the hand of a man,the man is wearing a black shirt,a laptop on a table,a laptop on a table,a white piece of paper\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man holding a laptop in his hands.",
            "dense_caption": "the opened laptop on the table,this is a laptop,a black box with wires plugged into it,a person is sitting on a wooden table,a black wrist band,a hand holding a laptop,laptop screen is on,the screen is white,laptop computer on desk,the keyboard of a laptop,the hand of a person,the brown and black flip phone\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
        {
            "action": "unboxing",
            "abstract_caption": "a man holding a laptop in his hands.",
            "dense_caption": "a man playing the wii,a black framed projector screen,a silver laptop,a person is holding a wii controller,a white apple computer,black cords on the desk,a man with brown hair\n",
        }
    ]
    command2 = ["when the guy is focusing on something else than camera"]
    response2 = ListElements.get_instance(
        indexes=[1, 2],
        explanations=[
            "actions matched: looking at a large flat screen", 
            "matching actions: looking at screen",
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

def get_temporal_visual_prompt_llm(partial_variables={}, examples = []):
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
        prefix=PREFIX_TEMPORAL_VISUAL_PROMPT,
        suffix=SUFFIX_TEMPORAL_VISUAL_PROMPT,
        input_variables=["context", "metadata", "command"],
        partial_variables=partial_variables,
    )


def get_temporal_visual_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}\nVisual descriptions of 10-second clips: {metadata}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_TEMPORAL_VISUAL_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}\nVisual descriptions of 10-second clips: {metadata}"),
        ]
    )

    return final_prompt