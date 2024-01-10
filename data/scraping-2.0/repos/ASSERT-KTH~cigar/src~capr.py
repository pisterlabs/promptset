import logging

import openai
from src.chatgpt import ChatGPT
from src.framework import Framework
from src.bug import Bug
from src.prompts import Prompts as prompts
from prog_params import ProgParams as prog_params

class Capr(object):
    def __init__(self, chatgpt: ChatGPT, framework: Framework):
        self.name = "Capr"
        self.chatgpt = chatgpt
        self.framework = framework

    def repair(self, bug: Bug, mode: str, n_shot_count=1, sample_per_try=1, max_conv_length=3, max_tries=1):
        assert mode in ["SL", "SH", "SF"]
        n_shot_bugs=self.framework.get_n_shot_bugs(n=n_shot_count, bug=bug, mode=mode)

        plausible_patches = []
        plausible_patch_diffs = []
        first_plausible_patch_try = 0
        current_conversation_length = 0
        current_tries = 0
        total_cost = 0
        err_tf = 0
        err_ce = 0
        prefix = f"{self.framework.name}_{bug.project}_{bug.bug_id}_{mode}"

        while (current_tries < max_tries and len(plausible_patches) == 0):
            current_conversation_length = 0
            prompt = prompts.construct_initial_prompt(bug=bug, mode=mode, n_shot_bugs=n_shot_bugs)

            while (current_conversation_length < max_conv_length and current_tries < max_tries):
                current_tries += 1
                current_conversation_length += 1

                logging.info(f"Searching for plausible patch in {bug.project}-{bug.bug_id} ({mode}), try {current_tries} (ccl: {current_conversation_length})")
                try:
                    response, cost = self.chatgpt.call(prompt, num_of_samples=sample_per_try, prefix=f"{prefix}_{current_tries}")
                    response = response[0]
                except openai.error.InvalidRequestError as e:
                    logging.info(e)
                    err_ce += 1 # Count token exceeded limit as error
                    total_cost += prog_params.gpt35_model_token_limit # Exceeded Token limit
                    continue

                total_cost += cost

                patch = self.extract_patch_from_response(response)
                logging.debug(f"Validating response of {bug.project}-{bug.bug_id} ({mode})")
                test_result, result_reason, patch_diff = self.framework.validate_patch(bug=bug, proposed_patch=patch, mode=mode)

                if test_result == "PASS":
                    plausible_patches.append(patch)
                    plausible_patch_diffs.append(patch_diff)
                    first_plausible_patch_try = current_tries
                    logging.debug(f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) patch passed all tests")
                    break
                elif result_reason == bug.test_error_message:
                    feedback = prompts.test_fail_feedback()
                    logging.debug(f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) failed with same error message as original bug")
                else:
                    feedback = prompts.construct_feedback_prompt(test_result, result_reason, mode)
                    logging.debug(f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) failed with a different error message than original bug")

                if test_result == "FAIL":
                    err_tf += 1
                elif test_result == "ERROR":
                    err_ce += 1
                
                prompt.append({"role": "assistant", "content": f"""{response}"""})
                prompt.append(feedback)
        
        if len(plausible_patches) != 0 and not prog_params.stop_on_first_plausible_patch:
            while (current_tries < max_tries):
                current_tries += 1

                logging.info(f"Attempt to generate multiple plausible patches in {bug.project}-{bug.bug_id} ({mode}), try {current_tries} (pps: {len(plausible_patches)})")
                prompt = prompts.construct_plausible_path_prompt(bug, plausible_patches, mode)

                try:
                    response, cost = self.chatgpt.call(prompt, num_of_samples=sample_per_try, prefix=f"{prefix}_{current_tries}")
                    response = response[0]
                except openai.error.InvalidRequestError as e:
                    logging.info(e)
                    err_ce += 1 # Count token exceeded limit as error
                    total_cost += prog_params.gpt35_model_token_limit # Exceeded Token limit
                    break

                total_cost += cost

                patch = self.extract_patch_from_response(response)

                test_result, result_reason, patch_diff = self.framework.validate_patch(bug=bug, proposed_patch=patch, mode=mode)
                if test_result == "PASS" and patch not in plausible_patches:
                    plausible_patches.append(patch)
                    plausible_patch_diffs.append(patch_diff)

                if test_result == "FAIL":
                    err_tf += 1
                elif test_result == "ERROR":
                    err_ce += 1
        
        return plausible_patches, plausible_patch_diffs, total_cost, first_plausible_patch_try, current_conversation_length, current_tries, err_tf, err_ce, current_tries
    
    def extract_patch_from_response(self, response):

        if "```java" in response:
            patch = response[response.find("```java")+len("```java")+1:]
            patch = patch[:patch.find("\n```")]
        else:
            patch = response

        return patch
    