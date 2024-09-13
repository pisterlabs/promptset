import asyncio
import json
import os
import time
from errorCorrection_async import errorCorrection_opro
from translations_async import translation_opro
from summarization_async import summarization_opro
from QARefinement_async import qarefinement_opro

class OPRO:
    def __init__(self, openai_api_key) -> None:
        self.openai_api = openai_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api
    
    async def optimize(self, prompt, category, id=time.time()):
        opro_func = None
        match category:
            case "translation":
                opro_func = translation_opro
            case "error_correction":
                opro_func = errorCorrection_opro
            case "QA_refinement":
                opro_func = qarefinement_opro
            case "summarization":
                opro_func = summarization_opro
            case _:
                raise ValueError(f"Invalid category: {category}")

        # Return Optimized Prompt
        return {
                "ID": id,
                **await opro_func(
                    prompt,
                    f"{str(id)}",
                    PROMPTS_PER_STEP=20,
                    TRAINING_SAMPLE_SIZE=30,
                    TESTING_SAMPLE_SIZE=70,
                    STEP_COUNT=10,
                    MAX_PROMPT_SCORE_PAIRS=10,
                ),
                "category": category,
            }