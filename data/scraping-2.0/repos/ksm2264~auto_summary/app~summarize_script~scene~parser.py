from typing import List

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from app.summarize_script.models import SceneBuffer

scene_summary_template='''
You identify scene changes, and summarize scenes.
A scene change is simply an entrace or exit of a character.
A character speaking for the first time without a descripition of their entrance should be treated as them having been there already.

E.g.
Jen is sitting alone in a coffee shop. 
Mike then enters and asks her something. 
Mike then leaves and 
Jen says something to herself. 
Then Sarah enters and talks to Jen

scene list would be:
[JEN] summary : jen sits alone
[JEN, MIKE] : mike asks jen
[JEN] : jen talks to herself
[JEN, SARAH] : sarah talks to jen

Assume that currently these characters are present, so that would be your first scene.
{current_scene}

Next chunk:
{text_chunk}


Make sure the summary is short (maximum 4 words).
Make a scene change ONLY occurs when a character enters or exists. So no new scene when the current character roster is the same.

Format like this:
{format_instructions}

'''

scene_buffer_parser = PydanticOutputParser(pydantic_object=SceneBuffer)
prompt = PromptTemplate(
    template=scene_summary_template,
    input_variables = ['current_scene', 'text_chunk'],
    partial_variables={"format_instructions": scene_buffer_parser.get_format_instructions()}
)

smart_llm = ChatOpenAI(model_name='gpt-4')
dumb_llm = ChatOpenAI(model_name='gpt-3.5-turbo')
scene_buffer_generator = LLMChain(llm=smart_llm, prompt=prompt, verbose=False)
scene_buffer_fixer = OutputFixingParser.from_llm(llm = smart_llm, parser = scene_buffer_parser)


def get_scene_buffer(text_chunk: str, current_scene: List[str]) -> SceneBuffer:

    raw_response = scene_buffer_generator.predict(
        current_scene = current_scene,
        text_chunk = text_chunk
    )

    scene_buffer = scene_buffer_fixer.parse(raw_response)

    return scene_buffer