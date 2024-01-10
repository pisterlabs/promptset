from tqdm import tqdm
from openai.types.chat import ChatCompletion

from gpt_gleam.costs import CompletionUsageEstimator


class ChatCompletionProgress(tqdm):
    def __init__(self, total: int, seen: int, *args, **kwargs):
        super().__init__(total=total, initial=seen, *args, **kwargs)
        self.chat_usage = CompletionUsageEstimator()
        self.chat_usage.init(total - seen)

    def __enter__(self):
        return super().__enter__()

    def update(self, completion: ChatCompletion) -> bool | None:
        self.chat_usage.update(completion)
        res = super().update(1)
        use = self.chat_usage.estimate()
        self.set_postfix(
            {
                "cost": f"${use.total_cost:.2f}",
            }
        )
        return res
