import json
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from promptTemplate import get_seq_as_dict

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# llm = ChatOpenAI(model_name="gpt-4") # type: ignore
llm = ChatOpenAI() # type: ignore


comp_template = """
You are a powerful AI Assistant which is well trained on many videos.

The input will contain sequence_input which is a JSON that contains the sequence configuration.
Based on the generatedText key, the Assistant will generate a new JSON that contains the sequence configuration based on the template below.



The template for the new JSON to be created is as follows:

{{
    "sequenceNo": 3, // The sequence number from the sequence input
    "from": 600, // From the sequence_input
    "durationInFrames": 300, // From the sequence_input
    "transition": {{
    "type": "in" // From the sequence_input
    }},
    "structure": {{
    "backgroundColor": "#000000", // Any color that suits this sequence to be choosen by the assistant so that it looks good with the components generated
    "useTemplate": false, // From the sequence_input
    "template": null, // From the sequence_input
    "generatedText": "Cat has a soft body", // The text that will be generated for this sequence
    "components": [] // List of components generated for this sequence. The components are generated based on the generatedText. The components can be text, paragraph, image. The assistant chooses a combination of components that looks good for the sequence. The components that can be used are mentioned in the instructions that follow this. There can be multiple components in a sequence.
    }}
}}

Components that can be used are:
1. paragraph
2. image
3. text

1. The template for the new JSON for paragraph component to be created is as follows:
{{
  "name": "Card_1_Paragraph", // Name of the paragraph
  "type": "paragraph", // Type of the component. Should be paragraph for this paragraph component
  "text": "Lorem ipsum dolor sit amet", // This is based on generatedText from the sequence_input
  "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
  "wordByWord": true, // Assisant can choose to show the text word by word or all at once
  "wordByWordSpeed": 2, // Speed at which the words are shown. This is based on the durationInFrames from the sequence_input. Assistant chooses a value between 1 and 5.
  "style": {{ // Assistant chooses a style for the paragraph component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
    "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
    "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
    "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
    "left": "167px", // Assistant chooses a left for the component. This is CSS left.
    "top": "660px", // Assistant chooses a top for the component. This is CSS top.
    "fontFamily": "Nunito", // Assistant chooses a fontFamily for the component. This is CSS fontFamily.
    "fontStyle": "normal", // Assistant chooses a fontStyle for the component. This is CSS fontStyle. Possible values are normal, italic.
    "fontWeight": 400, // Assistant chooses a fontWeight for the component. This is CSS fontWeight. Possible values are 400, 500, 600, 700, 800, 900.
    "fontSize": "24px", // Assistant chooses a fontSize for the component. This is CSS fontSize.
    "lineHeight": "29px", // Assistant chooses a lineHeight for the component. This is CSS lineHeight.
    "display": "flex", // Assistant chooses a display for the component. This is CSS display.
    "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
    "textAlign": "center",  // Assistant chooses a textAlign for the component. This is CSS textAlign.
    "color": "#ffffff", // Assistant chooses a color for the component. This is CSS color.
    "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
    "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
  }},
  "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
  "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
    {{
      "type": "entry",
      "frameFrom": 60,
      "valueFrom": 200,
      "valueTo": 0
    }},
    {{
      "type": "entryTranslateY",
      "frameFrom": 60,
      "valueFrom": -100,
      "valueTo": 0
    }}
  ]
}}

2. The template for the new JSON for image component to be created is as follows:
{{
    "name": "Card_1_Image", // Name of the image
    "type": "image", // Type of the component. Should be image for this image component
    "src": "https://images.unsplash.com/photo-1611095772760-5b6b5b9d0b0f?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Y2F0JTIwYmFyfGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&w=1000&q=80", // This is based on generatedText from the sequence_input
    "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
    "style": {{ // Assistant chooses a style for the image component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
        "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
        "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
        "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
        "left": "167px", // Assistant chooses a left for the component. This is CSS left.
        "top": "660px", // Assistant chooses a top for the component. This is CSS top.
        "display": "flex", // Assistant chooses a display for the component. This is CSS display.
        "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
        "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
        "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
    }},
    "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
    "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
        {{
            "type": "entry",
            "frameFrom": 60,
            "valueFrom": 200,
            "valueTo": 0
        }},
        {{
            "type": "entryTranslateY",
            "frameFrom": 60,
            "valueFrom": -100,
            "valueTo": 0
        }}
    ]
}}

3. The template for the new JSON for text component to be created is as follows: 
{{
    "name": "Card_1_Text", // Name of the text
    "type": "text", // Type of the component. Should be text for this text component
    "text": "Cat has a soft body", // This is based on generatedText from the sequence_input
    "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
    "style": {{ // Assistant chooses a style for the text component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
        "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
        "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
        "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
        "left": "167px", // Assistant chooses a left for the component. This is CSS left.
        "top": "660px", // Assistant chooses a top for the component. This is CSS top.
       "fontFamily": "Nunito", // Assistant chooses a fontFamily for the component. This is CSS fontFamily.
    "fontStyle": "normal", // Assistant chooses a fontStyle for the component. This is CSS fontStyle. Possible values are normal, italic.
    "fontWeight": 400, // Assistant chooses a fontWeight for the component. This is CSS fontWeight. Possible values are 400, 500, 600, 700, 800, 900.
    "fontSize": "24px", // Assistant chooses a fontSize for the component. This is CSS fontSize.
    "lineHeight": "29px", // Assistant chooses a lineHeight for the component. This is CSS lineHeight.
    "display": "flex", // Assistant chooses a display for the component. This is CSS display.
    "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
    "textAlign": "center",  // Assistant chooses a textAlign for the component. This is CSS textAlign.
    "color": "#ffffff", // Assistant chooses a color for the component. This is CSS color.
    "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
    "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
    }},
    "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
    "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
        {{
            "type": "entry",
            "frameFrom": 60,
            "valueFrom": 200,
            "valueTo": 0
        }},
        {{
            "type": "entryTranslateY",
            "frameFrom": 60,
            "valueFrom": -100,
            "valueTo": 0
        }}
    ]
}}


sequence_input :
{sequence_input}

output:


"""

seq_output = get_seq_as_dict(video_input='', user_query="Generate a video of duration 900 frames in 30 fps with width: 1080 and height: 1920. The video is about five facts about cats.")

print(seq_output)

# Iterate over the sequences and generate the video
for seq in seq_output['videoInput']['sequences']:
    print(f"Sequence No: {seq['sequenceNo']}")
    print(f"Sequence: {seq}")
    # Convert seq dict to json
    seq_json = json.dumps(seq)
    # Seq_json to string
    seq_json_str = str(seq_json)
    comp_prompt_template = PromptTemplate(input_variables=["sequence_input"], template=comp_template)
    comp_llm_chain = LLMChain(prompt=comp_prompt_template, llm=llm)
    comp_output = comp_llm_chain.run(sequence_input=seq_json_str)
    print(f"comp_output: {comp_output}")
    comp_output_json = json.loads(comp_output)
    print(f"comp_output_json: {comp_output_json}")
    # Convert comp_output_json to dict
    comp_output_dict = dict(comp_output_json)
    print(f"comp_output_dict: {comp_output_dict}")
    # Replace the sequence in the seq_output with the comp_output_dict
    seq_output['videoInput']['sequences'][seq['sequenceNo']-1] = comp_output_dict

# Print formatted seq_output
print(f"seq_output: {seq_output}")
    


