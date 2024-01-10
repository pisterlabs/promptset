import os

from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter
from img_to_text.minigpt4 import MiniGPT4Captioning
from img_to_text.minigpt4_hugging_face import MiniGPT4HuggingFaceInterface
from img_to_text.openai_interface import OpenAIInterface
from resources.resources import img_dir_path


class MiniGPT4AssistedGPT3(ImageToCaptionConverter):
    ERROR_PLACEHOLDER = 'Not Applicable'
    def __init__(self):
        self.mini_gpt4 = MiniGPT4Captioning(self.default_minigpt4_prompt())
        self.interface = OpenAIInterface()

    def default_prompt(self, description: str):
        assert '`' not in description
        return f'''
# SETTING:
You are part of an image processing pipeline. You will be given an automatically generated description of an image that is part of a video.
The goal of the pipeline is to derive some interesting information about that image and to display it for some seconds on the screen as an overlay on top of the video.
Your role in this pipeline is to provide some additional background information to be used as a caption for the video.
Keep in mind that the user will not see the result as an image but rather as a video with your text as an overlay.
Do not describe what objects look like or what is obviously visible in the image. Do not use phrases like "The image shows" or "The video displays" and no not literally quote the title.
 
# DESCRIPTION:
Here is the description (output of previous pipeline steps) that needs to be summarized to a caption - it is enclosed in triple quotes:
```markdown
{description}
```

# OUTPUT FORMAT:
Regarding the caption format, please provide up to 3 bullet points with at most six words each (NOT whole sentences), where each bullet point contains some interesting scientific facts about the locations/animals/people shown in the image.
Such information can and should include facts that are not listed in the description above, for example if an animal is shown in the foreground, you could provide species name and behaviour.

An example output (for a different image that shows a tiger shark) could be:
- Tiger shark (Galeocerdo cuvier)
- Solitary, mostly nocturnal hunter
- Length: 3-6 m

If some part of the pipeline failed, (e.g. no image available), just answer with "{self.ERROR_PLACEHOLDER}" and in any case do not answer in full sentences.
'''

    def default_minigpt4_prompt(self):
        return '''
You are part of an image processing pipeline.
You are given an an image that is part of a video.
The goal of the pipeline is to derive some interesting information about that image and to display it for some seconds on the screen as an overlay on top of the video.
Your role in this pipeline is to summarize what is displayed in the image including some informative and interesting facts.
Your summary will then be used by a later pipeline component to create a caption for the video.
'''

    def _convert(self, img_data: ImageMetadata) -> str:
        minigpt4_output = self.mini_gpt4.cached_convert(img_data)
        prompt = self.default_prompt(minigpt4_output + '\n' + img_data.extra_info_string())
        gpt_caption = self.interface.send_prompt(prompt)
        if self.is_error_message(gpt_caption):
            return ''
        return gpt_caption

    def is_error_message(self, gpt_caption):
        return self.ERROR_PLACEHOLDER.lower() in gpt_caption.lower()
