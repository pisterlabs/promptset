from models.self_improve.base import LLMwVerifier, SelfImprove_GPT_QA
from prompts.logical_deduction import *
from langchain import PromptTemplate, LLMChain
from runners.utils import chain_run_wrapper

import re


TEMPLATE_LOGICAL_DEDUCTION_UPDATE = """
{examples}
Q: {question}
Answer: Let's think step by step. Let's think step by step. Let "??" represents 0 or more objects, and "?" represents exactly 1 object.
{attempted_answer} [END]
Feedback: {feedback}
Updated Answer: Let's think step by step. Let's think step by step. Let "??" represents 0 or more objects, and "?" represents exactly 1 object.{prev_partial_answer}
""".strip()


class LLMwVerifier_LogicalDeduction(LLMwVerifier):
    def __init__(
            self,
            init_ans_model,
            improve_model,
            verifier_model,
            tokenizer=None,
            max_updates=5,
            verbose=False,
            save_log=True,
            is_eval=False):
        self.init_ans_model = init_ans_model  # used for the first answer, e.g.used for generating corrupted first step
        self.improve_model = improve_model  # all answers except the first one
        self.verifier_model = verifier_model  # a llm verifier
        self.tokenizer = tokenizer  # in case verifier is neural
        self.verbose = verbose
        self.save_log = save_log
        self.is_eval = is_eval  # is_eval = True when testing eval_llm_self_improve.py

        self.prompt_update = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "feedback", "prev_partial_answer"],
            template=TEMPLATE_LOGICAL_DEDUCTION_UPDATE,
        )
        self.chain_get_update = LLMChain(
            llm=self.improve_model.llm,
            prompt=self.prompt_update,
            output_key="updated_answer",
            verbose=verbose
        )

        # other metadata
        self.num_updated = 0
        self.max_updates = max_updates
        self.logs: dict = {}
        return
    
    def get_verifier_feedback(self, llm_output):
        attempt = llm_output['attempted_answer'].strip()
        correct_answer = llm_output['correct_answer']
        
        last_step = attempt.split('\n')[-1]
        choice_outputs = re.findall(r'the answer is \(([a-zA-Z])\)', last_step)
        if len(choice_outputs) != 1:
            return "[ERROR]"
        final_chioce = choice_outputs[0]
        
        num_steps = len(attempt.split('\n')) - 1
        if not self.is_eval and correct_answer.lower() == final_chioce.lower():
            return f"Step (1) to step ({num_steps}) are correct. The final response is also correct."

        # otherwise, generate feedback
        output = self.verifier_model.generate(llm_output, out_dict=True)
        generated_feedback = output['feedback']
        generated_feedback = self._post_process_feedback(generated_feedback)
        if self.verbose:
            print("Original feedback:", output['feedback'])
            print("Cleaned feedback:", generated_feedback)
        if not self._check_feedback_format_for_bad_rationale(generated_feedback):
            return "[ERROR] " + generated_feedback
        return generated_feedback
    
    def _to_log_key(self, input_data):
        question = input_data['question'].strip()
        formatted_choices = input_data['formatted_choices'].strip()
        formatted = f"""
        Q: {question}
        Options:
        {formatted_choices}
        """.replace('    ', '').strip()
        if formatted in self.logs:
            print(formatted)
            raise ValueError("Duplicate question")
        return formatted

    def generate(self, input_data, **gen_kwargs):
        updated = 0
        log_key = self._to_log_key(input_data)

        llm_output = self.init_ans_model.generate(input_data, out_dict=True)
        llm_output['attempted_answer'] = llm_output.pop('init_answer').strip()
        llm_output['correct_answer'] = input_data['answer'].strip()  # used by verifier

        # TODO: compatibility issue: code written for LLmEditor perform this inside editors
        llm_output['question'] = f'{input_data["question"]}\nOptions:\n{input_data["formatted_choices"]}'.strip()

        legit_update = self._check_attempted_answer(llm_output['attempted_answer'], '')
        if legit_update:
            feedback = self.get_verifier_feedback(llm_output)
            if self.verbose:
                print("Feedback:", feedback)
            done = self.check_if_done(feedback)
        else:
            feedback = "[ERROR]"
            done = True

        # logging
        if self.save_log:
            self._save_log(log_key, {
                'updated': updated,
                'llm_output': llm_output,
                'feedback': feedback,
                'done': done
            })
        
        if done:
            return llm_output['attempted_answer']
        
        self.num_updated += 1
        while not done and updated < self.max_updates:
            update_prompt = {
                **llm_output,
                "feedback": feedback,
                "prev_partial_answer": "",
                "examples": EXAMPLES_LOGICAL_DEDUCTION_UPDATE  # EXAMPLES_UPDATE
            }
            llm_output = chain_run_wrapper(self.chain_get_update, update_prompt)
            # check feedback
            prev_answer = llm_output['attempted_answer']
            llm_output['attempted_answer'] = llm_output.pop('updated_answer').strip()
            llm_output['correct_answer'] = input_data['answer'].strip()

            legit_update = self._check_attempted_answer(llm_output['attempted_answer'], prev_answer)
            
            if legit_update:
                feedback = self.get_verifier_feedback(llm_output)
                if self.verbose:
                    print("Feedback:", feedback)
                done = self.check_if_done(feedback)
            else:
                feedback = "ERROR"
                done = True
            
            if self.save_log:
                self._save_log(log_key, {
                    'updated': updated,
                    'llm_output': llm_output,
                    'feedback': feedback,
                    'done': done
                })
            updated += 1
        return llm_output['attempted_answer'].strip()


class SelfImprove_GPT_LogicalDeduction(SelfImprove_GPT_QA):
    def __init__(self, model, tokenizer, manual_prompt=False, input_max_length=1024, max_new_tokens=1024, additional_info=" Let's think step by step.", gen_kwargs={}):
        super().__init__(model, tokenizer, manual_prompt, input_max_length, max_new_tokens, additional_info, gen_kwargs=gen_kwargs)