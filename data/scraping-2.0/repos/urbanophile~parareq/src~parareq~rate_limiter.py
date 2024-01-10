import time
from queue import Queue
import random

# import openai

# import data classes
from dataclasses import dataclass, field

# class JobDispatcher:
#     def __init__(openai_key, max_jobs=10):
#         self.max_jobs = max_jobs
#         openai.api_key = openai_key

#     def run_job(self,args):
#         openai.ChatCompletion.create(**args)


@dataclass
class Job:
    id: int
    function: callable
    args: dict
    result: any = None
    error: Exception = None
    status: str = "pending"

    def to_dict(self):
        return {
            "id": self.id,
            "function": self.function.__name__,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "status": self.status,
        }


class RateLimiter:
    def __init__(self, rate, per):
        self.rate = rate
        self.per = per
        self.allowance = rate

    def update_allowance(self, time_passed):
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate  # throttle

    def is_limited(self):
        return self.allowance < 1.0

    def update_usage(self):
        self.allowance -= 1.0


class TokenLimiter(RateLimiter):
    def is_limited(self):
        return super().is_limited()

    def update_usage(self):
        super().update_usage()


class JobManager:
    def __init__(self, call_rate, call_per, token_rate, token_per, total_timeout):
        self.token_limiter = RateLimiter(token_rate, token_per)
        self.call_limiter = RateLimiter(call_rate, call_per)

        self.last_check = time.time()  # Unit: seconds
        self.todo_queue = Queue()
        self.total_timeout = total_timeout
        self.func = example_task
        self.results = []

    def now(self):
        return time.time()

    def add_job(self, job_params):
        print(f"adding_task: {job_params}")
        self.todo_queue.put(job_params)

    def attempt_job(self, job):
        current = self.now()
        time_passed = current - self.last_check
        self.last_check = current

        self.token_limiter.update_allowance(time_passed)
        self.call_limiter.update_allowance(time_passed)

        if self.token_limiter.is_limited() or self.call_limiter.is_limited():
            self.todo_queue.put(job)  # Reinsert the limited request into the queue
        else:
            task_result = self.try_single_job(job)
            self.token_limiter.update_usage()
            self.call_limiter.update_usage()
            self.results.append(task_result.to_dict())

    def try_single_job(self, job: Job):
        try:
            job.result = job.function(job.args)
            job.status = "success"
        except Exception as e:
            print(f"error: {e}")
            job.error = e
            job.status = "failed"
        return job

    def start_jobs(self):
        self.run()

    def run(self):
        while not self.todo_queue.empty():
            task = self.todo_queue.get()
            self.attempt_job(task)
        print(self.results)


def main_test():
    # Example usage:
    manager = JobManager(
        call_rate=5.0,
        call_per=8.0,
        token_rate=5.0,
        token_per=8.0,
        total_timeout=10.0,
    )

    # Simulate task queuing
    for task_desc in range(10):
        job = Job(id=task_desc, args={"task_desc": task_desc}, function=example_task)
        manager.add_job(job)

    # Simulate task processing from the queue
    manager.start_jobs()


def main_prod():
    # Example usage:
    manager = JobManager(
        call_rate=5.0,
        call_per=8.0,
        token_rate=5.0,
        token_per=8.0,
        total_timeout=10.0,
    )

    # Simulate task queuing
    for task_desc in range(10):
        job = Job(id=task_desc, args={"task_desc": task_desc}, function=example_task)
        manager.add_job(job)

    # Simulate task processing from the queue
    manager.start_jobs()


if __name__ == "__main__":
    main_test()
