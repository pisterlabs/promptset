from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch
import time

llm_chain = None

class FalconWrapper:
    def release_zombie_memory():
        for proc in psutil.process_iter(["cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if len(cmdline) == 2 and cmdline[0] == "python":
                    if proc.status == psutil.STATUS_ZOMBIE:
                        proc.terminate()
                        proc.wait()
            except psutil.Error:
                pass

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from langchain import PromptTemplate, LLMChain

        import os
        os.environ["SWAP_ENABLED"] = "True"
        os.environ["SWAP_THRESHOLD"] = "9"
        import gc

        gc.collect()

        import torch

        torch.cuda.empty_cache()

        model = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)

        import torch
        import transformers
        
        self.release_zombie_memory

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        time.sleep(5)       # process is crashed due to RAM so trying to stop the process to wait the memory clear
        # Create the pipe schedular
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # self.pipe = pipe.to("cuda")

        from langchain import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipeline)

        template = """Question: {question}
        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

 

            
    def generate_query_response(self, text_prompt: str):

        question = text_prompt
        result = self.llm_chain.run(question)
        print(f"Warning: Result ->>>> {result}")

    
        return result
