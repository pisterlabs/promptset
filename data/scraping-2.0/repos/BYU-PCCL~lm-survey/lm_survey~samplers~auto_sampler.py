from lm_survey.samplers.hf_sampler import HfSampler
from lm_survey.samplers.openai_sampler import OpenAiSampler
from lm_survey.samplers.base_sampler import BaseSampler


class AutoSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_name.startswith("gpt3") or self.model_name.startswith("gpt4"):
            self.sampler = OpenAiSampler(*args, **kwargs)
        else:
            self.sampler = HfSampler(*args, **kwargs)

    def rank_completions(self, prompt, completions):
        return self.sampler.rank_completions(prompt, completions)

    def send_prompt(self, prompt, n_probs):
        return self.sampler.send_prompt(prompt, n_probs)

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        return self.sampler.sample_several(prompt, temperature, n_tokens)


if __name__ == "__main__":
    sampler = AutoSampler("gpt3-ada")
    text = sampler.get_best_next_token(
        prompt="What is the capital of France?\nThe capital of France is",
    )
    print(text)
