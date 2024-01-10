from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from openai_ratelimiter import ChatCompletionLimiter

model_name = "gpt-3.5-turbo-16k"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Morocco."},
]


class Executor(ThreadPoolExecutor):
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.shutdown(wait=False, cancel_futures=True)


def test_TPM():
    max_tokens = 200
    chatlimiter = ChatCompletionLimiter(
        model_name=model_name,
        RPM=3_000,  # we will ignore this by setting a high value, we will make another test to test this.
        TPM=1_125,  # 1_125 = 225 * 5
        redis_host="localhost",
        redis_port=6379,
    )
    chatlimiter.clear_locks()
    with Executor(max_workers=1) as executor:
        for _ in range(5):
            future = executor.submit(
                chatlimiter.limit(messages=messages, max_tokens=max_tokens).__enter__
            )
            if 0 <= _ <= 4:
                try:
                    future.result(timeout=2)
                except TimeoutError:
                    pytest.fail("The request should have been completed.")
                continue
        if not chatlimiter.is_locked(messages=messages, max_tokens=max_tokens):
            pytest.fail("The request should have timed out.")
