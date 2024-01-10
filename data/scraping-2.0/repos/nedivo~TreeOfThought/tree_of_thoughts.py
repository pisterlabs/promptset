import heapq
import logging
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class TreeOfThoughts:
    def __init__(self, model, max_depth, breadth_limit, initial_problem):
        self.model = model
        self.max_depth = max_depth
        self.breadth_limit = breadth_limit
        self.initial_problem = initial_problem
        self.client = OpenAI()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.max_requests_per_minute = (
            6  # change this only if you know they should be different
        )
        self.max_tokens_per_minute = (
            40000  # change this only if you know they should be different
        )
        self.available_request_capacity = self.max_requests_per_minute
        self.available_token_capacity = self.max_tokens_per_minute
        self.last_update_time = time.time()

        self.logger.debug("Initialized")

    def openai_chat_completion(self, messages):
        self.logger.debug("OpenAI Chat Completion")
        self.wait_for_rate_limit()
        responses = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        results = [
            choice.message.content.strip()
            for choice in responses.choices
            if choice.message.role == "assistant"
        ]
        return results

    def wait_for_rate_limit(self):
        self.logger.debug("Waiting for rate limit availability")
        while True:
            current_time = time.time()
            seconds_since_update = current_time - self.last_update_time

            self.available_request_capacity = min(
                self.available_request_capacity
                + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute,
            )
            self.available_token_capacity = min(
                self.available_token_capacity
                + self.max_tokens_per_minute * seconds_since_update / 60.0,
                self.max_tokens_per_minute,
            )

            if (
                self.available_request_capacity >= 1
                and self.available_token_capacity >= 1000
            ):  # Assuming 1000 tokens per request
                self.available_request_capacity -= 1
                self.available_token_capacity -= (
                    1000  # Deduct estimated tokens used per request
                )
                self.last_update_time = current_time
                self.logger.debug(
                    "Available request capacity: %s, token capacity: %s",
                    self.available_request_capacity,
                    self.available_token_capacity,
                )
                break

            time.sleep(1)

    def generate_thoughts(self, state):
        self.logger.debug("Generating thoughts for state: %s", state)
        last_thought = state.split(" -> ")[-1]
        messages = [
            {
                "role": "system",
                "content": f"""You are an assistant skilled in problem solving.
                    The initial problem is: {self.initial_problem}.""",
            },
            {
                "role": "user",
                "content": f"What are the next steps or alternative approaches for: {last_thought}?",
            },
        ]
        return self.openai_chat_completion(messages)

    def create_thought_prompt(self, state):
        return f"What are the potential next steps given the state: {state}?"

    def evaluate_states(self, states):
        self.logger.debug("Evaluate States")
        evaluated_states = []
        for state in states:
            value = self.evaluate_state(state)
            evaluated_states.append((value, state))
        self.logger.debug("Evaluated States: %s", evaluated_states)
        return evaluated_states

    def evaluate_state(self, state):
        self.logger.debug("Evaluate State")
        # self.logger.debug("Evaluate State: %s", state)
        new_state_part = state.split(" -> ")[-1]
        messages = [
            {
                "role": "system",
                "content": f"""You are an assistant skilled in evaluating problem-solving states.
                    The initial problem is: {self.initial_problem}.""",
            },
            {
                "role": "user",
                "content": f"Rate the effectiveness of this new approach: '{new_state_part}' on a scale from 1 to 10.",
            },
        ]
        responses = self.openai_chat_completion(messages)
        for response in responses:
            match = re.search(r"\d+", response)
            if match:
                try:
                    score = float(match.group())
                    self.logger.debug("Evalue State Return: %s", score)
                    return score
                except ValueError:
                    self.logger.error("Invalid evaluation format: %s", response)
            else:
                self.logger.error(
                    "No numeric evaluation found in response: %s", response
                )
        self.logger.debug("Evalue State Return Outside: 0")
        return 0

    def create_evaluation_prompt(self, state):
        self.logger.debug("Create Evaluation Prompt")
        return (
            f"Consider the following solution step: '{state}'. "
            "On a scale from 1 to 10, how effective is this step in solving the problem? "
            "Please provide a numerical evaluation."
        )

    def search(
        self, current_state, quality_threshold, max_iterations, progress_bar=None
    ):
        self.logger.debug("Search")
        self.logger.debug("Current State: %s", current_state)
        logs = []
        queue = [(0, current_state)]
        iteration_count = 0
        best_state = None  # initialize the best_state as None
        best_value = float("-inf")  # initialize the best_value as negative infinity

        for i in range(max_iterations):
            iteration_count += 1
            self.logger.debug(
                "Current depth: %s | Queue size: %s", iteration_count, len(queue)
            )

            if progress_bar is not None:
                progress = (i + 1) / max_iterations
                self.logger.debug("Progress Change %s", progress)
                progress_bar.progress(progress)

            next_level = []
            while queue:
                value, state = heapq.heappop(queue)
                thoughts = self.generate_thoughts(state)
                for thought in thoughts:
                    new_state = f"{state} -> {thought}"
                    next_level.append(new_state)

            evaluated_states = self.evaluate_states(next_level)
            queue = heapq.nlargest(
                self.breadth_limit, evaluated_states, key=lambda x: x[0]
            )
            if queue[0][0] > best_value:
                best_value, best_state = queue[0]

            if not queue or self.is_termination_condition_met(
                best_value, iteration_count, max_iterations, quality_threshold
            ):
                break

            logs.append(f"Finished iteration {i+1} with state: {best_state}")

        if progress_bar is not None:
            progress_bar.progress(100)

        return best_state.split(" -> ")[-1], logs

    def is_termination_condition_met(
        self, current_quality, iteration_count, max_iterations, quality_threshold
    ):
        # current_quality = self.evaluate_state(current_state)
        if current_quality >= quality_threshold:
            self.logger.debug(
                "Termination condition met: Quality threshold (%s) reached with current quality (%s)",
                quality_threshold,
                current_quality,
            )
            return True

        if iteration_count >= max_iterations:
            self.logger.debug(
                "Termination condition met: Maximum iterations (%s) reached at iteration %s",
                max_iterations,
                iteration_count,
            )
            return True

        return False
