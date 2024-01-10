import openai
from typing import Any, List
from datetime import datetime
from tqdm import tqdm
import pytorch_lightning as pl
from src.utils.eval_script import a2_evaluation, local_evaluation


class Gpt3(pl.LightningModule):
    def __init__(self, model, max_tokens, temperature, top_p, n, stream,
                 logprobs, presence_penalty, frequency_penalty, stop, api_key, output: str = 'logs/train_result', arg_finder=0):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.logprobs = logprobs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.output = output
        self.arg_finder = arg_finder
        try:
            self.local_eval = local_evaluation
        except:
            self.local_eval = True
        openai.api_key = api_key

    def forward(self, x: str):
        return openai.Completion.create(model=self.model,
                                        prompt=x,
                                        max_tokens=self.max_tokens,
                                        temperature=self.temperature,
                                        top_p=self.top_p,
                                        n=self.n,
                                        stream=self.stream,
                                        logprobs=self.logprobs,
                                        presence_penalty=self.presence_penalty,
                                        frequency_penalty=self.frequency_penalty,
                                        stop=self.stop, )

    def debug_step(self, batch: any):
        x, y = batch
        return y.output_tokens, x, y

    def step(self, batch: Any):
        x, y = batch
        output_prompt = self.forward(x)
        return output_prompt.choices.text, x, y

    def training_step(self, batch: Any, batch_idx: int):
        #todo
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def validation_epoch_end(self, outputs: List[Any]):
        return -1

    def test_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def test_epoch_end(self, outputs: List[Any]):
        # todo safe all found events in a dict
        if self.local_eval:
            f1, prec, rec = local_evaluation(outputs, self.arg_finder)
        else:
            f1, prec, rec = a2_evaluation(outputs, self.output, self.arg_finder)
        self.log("val/f1", f1, on_epoch=True)
        self.log("val/precision", prec, on_epoch=True)
        self.log("val/recall", rec, on_epoch=True)


    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
