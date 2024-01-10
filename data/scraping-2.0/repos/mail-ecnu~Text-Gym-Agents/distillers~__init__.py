from .raw_prompt_generator import RawPromptGenerator
from .self_reflection import RefletionGenerator
from .traj_prompt_summarizer import TrajPromptSummarizer
from .guider import Guidance_Generator

REGISTRY = {}
REGISTRY['raw_distiller'] = RawPromptGenerator
REGISTRY['reflect_distiller'] = RefletionGenerator
REGISTRY['traj_distiller'] = TrajPromptSummarizer
REGISTRY['guide_generator'] = Guidance_Generator
