"""Tool for generating images."""
from steamship import Steamship,Block, Steamship, MimeTypes, File, Tag,DocTag #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.tools.speech_generation import GenerateSpeechTool #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task #upm package(steamship)
from steamship.agents.schema import AgentContext #upm package(steamship)
from transloadit import client as tlclient #upm package(pytransloadit)
import logging
from tools.active_companion import VOICE_ID #upm package(steamship)




class VoiceToolOGG(GenerateSpeechTool):
    """Tool to generate audio from text."""

    name: str = "GenerateSpokenAudio"
    human_description: str = "Generates spoken audio from text."
    agent_description: str = (
        "Use this tool to generate spoken audio from text, the input should be a plain text string containing the "
        "content to be spoken."        
    )

    prompt_template = (
        "{subject}"
    )
    def run(self, tool_input: List[Block], context: AgentContext,transloadit_api_key:str = "",transloadit_api_secret = "") -> Union[List[Block], Task[Any]]:
        if transloadit_api_key == "":
            print("API_KEY not set")
            logging.warning("Transloadit api key not set")
            block = Block(text="Transloadit api key not set")
            return [block]
            
        
        modified_inputs = [
            Block(text=self.prompt_template.format(subject=block.text))
            for block in tool_input
        ]
        speech = GenerateSpeechTool()
        speech.generator_plugin_config = {
            "voice_id": VOICE_ID
        }
        voice_result = speech.run(modified_inputs,context)
        voice_result[0].set_public_data(True)

        tl = tlclient.Transloadit(transloadit_api_key, transloadit_api_secret)
        assembly = tl.new_assembly()
        assembly.add_step("imported_postroll", "/http/import", {
        'url': voice_result[0].raw_data_url
        })

        assembly.add_step("ogg_encoded", "/audio/encode", {
        'use': 'imported_postroll',
        'result': True,
        'ffmpeg_stack': 'v4.3.1',
        'ffmpeg': {
            'q:a': -1,
            'b:a': 128000,
            'ar': 48000
        },
        'preset': 'opus'
        })

        assembly_response = assembly.create(retries=5, wait=True)
        #logging.info(assembly_response.data.get('assembly_ssl_url'))
        result = assembly_response.data['http_code']
        if result == 200:
            ogg_url = assembly_response.data['results']['ogg_encoded'][0]['ssl_url']
            block = Block(content_url=ogg_url,mime_type=MimeTypes.OGG_AUDIO,url=ogg_url)
            return [block]
        else:
            return[Block()]

if __name__ == "__main__":
    tool = VoiceToolOGG()
    with Steamship.temporary_workspace() as client:
        ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
