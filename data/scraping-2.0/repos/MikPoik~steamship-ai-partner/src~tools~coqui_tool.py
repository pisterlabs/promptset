"""Tool for generating audio"""
from steamship import Steamship, MimeTypes  #upm package(steamship)
from steamship.agents.llms import OpenAI  #upm package(steamship)
from steamship.agents.tools.speech_generation import GenerateSpeechTool  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task  #upm package(steamship)
from steamship.agents.schema import AgentContext
from steamship.agents.schema import AgentContext, Tool  #upm package(steamship)
import logging
import re
from transloadit import client as tlclient  #upm package(pytransloadit)
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin  #upm package(steamship)

#from tools.active_companion import VOICE_ID #upm package(steamship)


class CoquiTool(Tool):
    """Tool to generate audio from text."""

    name: str = "GenerateSpokenAudio"
    human_description: str = "Generates spoken audio from text."
    agent_description: str = (
        "Use this tool to generate spoken audio from text, the input should be a plain text string containing the "
        "content to be spoken.")

    prompt_template = ("{subject}")
    generator_plugin_handle: str = "coqui-tts"
    generator_plugin_config: dict = {
        "coqui_api_key": "",
        "language": "en",
        "speed": 1.0
    }
    tl_api_key = ""
    tl_api_secret = ""

    def run(self,
            tool_input: List[Block],
            context: AgentContext,
            api_key="",
            api_secret="") -> Union[List[Block], Task[Any]]:

        if api_key != "" and api_secret != "":
            self.tl_api_key = api_key
            self.tl_api_secret = api_secret

        audio_links = []

        def process_and_call_tool(batch_text):
            generator = context.client.use_plugin(
                self.generator_plugin_handle,
                config=self.generator_plugin_config,
                version="0.0.4")

            pattern = r'\*([^*]+)\*'  #Remove *gesture* texts from
            batch_text = re.sub(pattern, '', batch_text)

            task = generator.generate(text=batch_text,
                                      make_output_public=True,
                                      append_output_to_file=True)
            task.wait()

            #add link to concatenate list
            audio_links.append(task.output.blocks[0].raw_data_url)
            return task.output.blocks

        output_blocks = []

        meta_voice_id = context.metadata.get("instruction", {}).get("voice_id")
        if meta_voice_id is not None:
            if meta_voice_id != "none":
                self.generator_plugin_config["voice_id"] = meta_voice_id

        text_input = ""
        for block in tool_input:
            if not block.is_text():
                continue
            text_input += block.text

        words = text_input.split()
        current_batch = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_batch) > 250:
                # If adding the current word exceeds the batch size, process the current batch
                batch_text = ' '.join(current_batch)
                output_blocks.extend(process_and_call_tool(batch_text))
                current_batch = []
                current_length = 0

            current_batch.append(word)
            current_length += word_length

        # Process the last batch
        if current_batch:
            batch_text = ' '.join(current_batch)
            output_blocks.extend(process_and_call_tool(batch_text))

        #if Coqui produces multiple blocks, concatenate them
        if len(output_blocks) > 1:
            tl = tlclient.Transloadit(self.tl_api_key, self.tl_api_secret)
            #Concatenate audio files here and return the resulted mp3 as output block
            assembly = tl.new_assembly()
            # Import the generated audio files
            for i, url in enumerate(audio_links, start=1):
                assembly.add_step(f'import{i}', '/http/import', {'url': url})
            # Concatenate the imported audio files.
            assembly.add_step(
                'concat', '/audio/concat', {
                    "use": {
                        "steps": [{
                            'name': f'import{i}',
                            'as': f'audio_{i}'
                        } for i in range(1,
                                         len(audio_links) + 1)]
                    },
                    "result": {
                        "mp3": True
                    }
                })
            # execute assembly
            assembly_response = assembly.create(retries=5, wait=True)
            # get the concatenated audio file's url
            #print(assembly_response.data.get('assembly_ssl_url'))
            concat_file = assembly_response.data['results']['concat'][0][
                'ssl_url']
            #print(concat_file)

            #Import file with file importer
            file_importer = FileImporterMixin(client=context.client)
            imported_file = file_importer.import_url(concat_file)
            imported_file.set_public_data(True)
            #create block from file
            result_block = Block(content_url=imported_file.raw_data_url,
                                 mime_type=MimeTypes.MP3,
                                 url=imported_file.raw_data_url)
            #print(result_block.url)
            output_blocks = [result_block]

        return output_blocks


if __name__ == "__main__":
    tool = CoquiTool()
    with Steamship.temporary_workspace() as client:
        ToolREPL(tool).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client)))
