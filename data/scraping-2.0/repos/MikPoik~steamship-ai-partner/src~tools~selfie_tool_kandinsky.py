"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task  #upm package(steamship)
from steamship.agents.llms import OpenAI  #upm package(steamship)
from steamship.agents.schema import AgentContext, Tool  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from tools.active_companion import *  #upm package(steamship)
import logging


class SelfieToolKandinsky(Tool):
  """Tool to generate images from text using"""

  rewrite_prompt = SELFIE_TEMPLATE_PRE + "{subject}" + SELFIE_TEMPLATE_POST

  name: str = "generate_image"
  human_description = (
      "Used to generate images from text prompts. Only utilize this tool when the user explicitly requests an image. "
      "When using this tool, the input should be a plain text string that describes,"
      "in detail, the desired image.")
  agent_description = (
      "Useful if you need to generate a picture or selfie of you. Use if the user has asked for a picture response. The input is the text describing in detail the image with comma separated keywords. The output is the generated picture."
  )
  generator_plugin_handle: str = "replicate-kandinsky"
  generator_plugin_config: dict = {"replicate_api_key": "r8"}

  def run(self,
          tool_input: List[Block],
          context: AgentContext,
          context_id: str = "",
          api_key="") -> Union[List[Block], Task[Any]]:
    """Run the tool. Copied from base class to enable generate-time config overrides."""
    if api_key != "":
      self.generator_plugin_config["replicate_api_key"] = api_key

    generator = context.client.use_plugin(self.generator_plugin_handle,
                                          config=self.generator_plugin_config)

    prompt = self.rewrite_prompt.format(subject=tool_input[0].text)
    #print(prompt)
    task = generator.generate(
        text=prompt,
        make_output_public=True,
        append_output_to_file=True,
        options={
            "num_inference_steps": 1,  #75
            "num_steps_prior": 1,  #25
            "height": 384,  #1024
            "width": 384  #768
        })
    task.wait()
    blocks = task.output.blocks
    output_blocks = []
    for block in blocks:
      output_blocks.append(block)
    return output_blocks


if __name__ == "__main__":
  tool = SelfieToolKandinsky()
  client = Steamship(workspace="partner-ai-dev2-ws")
  context_id = "test-uuuid-5"
  #with Steamship.temporary_workspace() as client:
  ToolREPL(tool).run_with_client(client=client,
                                 context=with_llm(llm=OpenAI(client=client)))
