from models.base import GenerativeModel
from models.wrappers import LLM_QA_Feedback
from models.verifier.word_sorting import WS_Verifier, Scripted_WordSort_Feedback
from models.self_improve.base import LLMwVerifier, SelfImprove_GPT_QA, SelfImprove_woFeedback_GPT_QA
from prompts.word_sorting import *
from langchain import PromptTemplate, LLMChain
from runners.utils import chain_run_wrapper

import re


TEMPLATE_WORDSORT_FEEDBACK = """
Q: {question}
Answer: Let's think step by step.
{attempted_answer}
Feedback:
""".strip()


TEMPLATE_WORDSORT_UPDATE = """
{examples}
Q: {question}
Answer: Let's think step by step.
{attempted_answer} [END]
Feedback: {feedback}
Updated Answer: Let's think step by step.{prev_partial_answer}
""".strip()


# # the case when verifier can be a neural network
# class LLMwVerifier_WordSorting(GenerativeModel):
#     def __init__(
#             self, 
#             llm_model: LLM_WordSorting, 
#             verifier_model, 
#             tokenizer=None,
#             max_updates=10,
#             verbose=False,
#             save_log=True):
#         self.llm_model = llm_model
#         self.verifier_model = verifier_model  #.cuda()
#         if not isinstance(self.verifier_model, WS_Verifier):
#             # a neural verifier
#             self.verifier_model.eval()
#             self.tokenizer = tokenizer
#         self.verbose = verbose
#         self.save_log = save_log

#         self.prompt_update = PromptTemplate(
#             input_variables=["examples", "question", "attempted_answer", "feedback", "prev_partial_answer"],
#             template=TEMPLATE_WORDSORT_UPDATE
#         )
#         self.chain_get_update = LLMChain(
#             llm=self.llm_model.llm,
#             prompt=self.prompt_update,
#             output_key="updated_answer",
#             verbose=verbose
#         )
#         self.prompt_feedback = PromptTemplate(
#             input_variables=["question", "attempted_answer"],
#             template=TEMPLATE_WORDSORT_FEEDBACK
#         )

#         # other metadata
#         self.num_updated = 0
#         self.max_updates = max_updates
#         self.logs: dict = {}
#         return
    
#     def _format_verifier_output(self, verifier_output):
#         return verifier_output
    
#     def get_verifier_feedback(self, llm_output):
#         if isinstance(self.verifier_model, WS_Verifier):
#             question = llm_output["question"]
#             rationale = llm_output["attempted_answer"]
#             target = llm_output['targets']
#             try:
#                 cleaned_verifier_output = self.verifier_model.verify_rationale({
#                     'rationale': rationale,
#                     'question': question,
#                     'target': target
#                 })
#             except:
#                 cleaned_verifier_output = "[ERROR] The final response is also correct."
#         else:
#             verifier_prompt = {
#                 k: llm_output[k] for k in self.prompt_feedback.input_variables
#             }
#             verifier_input = self.prompt_feedback.format(**verifier_prompt)
#             input_ids = self.tokenizer(
#                 verifier_input, 
#                 max_length=1024,
#                 truncation=True,
#                 return_tensors="pt"
#             ).input_ids.to("cuda")

#             verifier_output = self.verifier_model.generate(input_ids, max_length=256, num_beams=1, num_return_sequences=1)
#             decoded_verifier_output = self.tokenizer.decode(verifier_output[0], skip_special_tokens=True).strip()
#             cleaned_verifier_output = self._format_verifier_output(decoded_verifier_output)
#         return cleaned_verifier_output
    
#     def check_if_done(self, feedback):
#         return "final response is also correct" in feedback.lower()

#     def _save_log(self, log_key, data):
#         if log_key not in self.logs:
#             self.logs[log_key] = []
#         self.logs[log_key].append(data)
#         return
    
#     def _to_log_key(self, input_data):
#         question = '\n'.join(input_data['inputs'].split('\n')[:-1])
#         question = question.strip()
#         formatted = f"""
#         Q: {question}
#         """.replace('    ', '').strip()
#         return formatted

#     def _check_attempted_answer_format(self, attempted_answer, prev_answer):
#         if attempted_answer.strip() == prev_answer.strip():
#             return False
#         steps = attempted_answer.split("\n")
#         last_step = steps[-1].strip()
#         if not last_step.lower().startswith('(final response)'):
#             return False
#         for step in steps:
#             # started a new question by itself
#             if 'Q:' in step:
#                 return False
#         return True

#     def generate(self, input_data, **gen_kwargs):
#         updated = 0
#         log_key = self._to_log_key(input_data)

#         llm_output = self.llm_model.generate(input_data, out_dict=True)
#         llm_output['attempted_answer'] = llm_output.pop('init_answer').strip()
#         llm_output['targets'] = input_data['targets'][0].split()  # used by script based verifier

#         feedback = self.get_verifier_feedback(llm_output)
#         if self.verbose:
#             print("Feedback:", feedback)
#         done = self.check_if_done(feedback)

#         # logging
#         if self.save_log:
#             self._save_log(log_key, {
#                 'updated': updated,
#                 'llm_output': llm_output,
#                 'feedback': feedback,
#                 'done': done
#             })
#         if done:
#             return llm_output['attempted_answer']
        
#         self.num_updated += 1
#         while not done and updated < self.max_updates:
#             update_prompt = {
#                 **llm_output,
#                 "feedback": feedback,
#                 "examples": EXAMPLES_WORDSORT_UPDATE  # EXAMPLES_UPDATE
#             }
#             llm_output = chain_run_wrapper(self.chain_get_update, update_prompt)
#             # check feedback
#             prev_answer = llm_output['attempted_answer']
#             llm_output['attempted_answer'] = llm_output.pop('updated_answer').strip()
#             legit_update = self._check_attempted_answer_format(llm_output['attempted_answer'], prev_answer)
            
#             if legit_update:
#                 llm_output['targets'] = input_data['targets'][0].split()
#                 feedback = self.get_verifier_feedback(llm_output)
#                 if self.verbose:
#                     print("Feedback:", feedback)
#                 done = self.check_if_done(feedback)
#             else:
#                 feedback = "ERROR"
#                 done = True
            
#             if self.save_log:
#                 self._save_log(log_key, {
#                     'updated': updated,
#                     'llm_output': llm_output,
#                     'feedback': feedback,
#                     'done': done
#                 })
#             updated += 1
#         return llm_output['attempted_answer'].strip()


class LLMwVerifier_WordSorting(LLMwVerifier):
    def __init__(
            self, 
            init_ans_model,
            improve_model, 
            verifier_model, 
            tokenizer=None,
            max_updates=10,
            verbose=False,
            save_log=True,
            is_eval=False):
        self.init_ans_model = init_ans_model
        self.improve_model = improve_model
        self.verifier_model = verifier_model  #.cuda()
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.save_log = save_log
        self.is_eval = is_eval

        self.prompt_update = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "feedback", "prev_partial_answer"],
            template=TEMPLATE_WORDSORT_UPDATE
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
        if isinstance(self.verifier_model, Scripted_WordSort_Feedback):
            question = llm_output["question"]
            rationale = llm_output["attempted_answer"].strip()
            target = llm_output['correct_answer']
            try:
                generated_feedback = self.verifier_model.verify_rationale({
                    'rationale': rationale,
                    'question': question,
                    'target': target
                })
            except:
                generated_feedback = "[ERROR] The final response is also correct."
        elif isinstance(self.verifier_model, LLM_QA_Feedback):
            # llm based
            attempt = llm_output['attempted_answer'].strip()
            correct_answer = llm_output['correct_answer']
            
            second_last_step = attempt.split('\n')[-2]
            last_step = attempt.split('\n')[-1]
            if not last_step.endswith('.'):
                last_step += '.'

            found_sorting = re.findall(r'the answer is:(.*)\.', last_step)
            if len(found_sorting) != 1:
                return "[ERROR]"
            
            final_sorting = found_sorting[0].strip()
            final_sorted_words = [word.strip() for word in final_sorting.split()]
            
            if not self.is_eval and correct_answer == final_sorted_words:
                try:
                    step_num = re.search(r"^\((\d+)(\.\d+)*\)", second_last_step).group(0)
                    return f"Step (1) to step {step_num} are correct. The final response is also correct."
                except:
                    return '[ERROR] cannot find step num in second last step.'

            # otherwise, generate feedback
            output = self.verifier_model.generate(llm_output, out_dict=True)
            generated_feedback = output['feedback']
            generated_feedback = self._post_process_feedback(generated_feedback)
            if self.verbose:
                print("Original feedback:", output['feedback'])
                print("Cleaned feedback:", generated_feedback)
            if not self._check_feedback_format_for_bad_rationale(generated_feedback):
                return "[ERROR] " + generated_feedback
        else:
            raise NotImplementedError('Verifier model not supported.')
        return generated_feedback
    
    def generate(self, input_data, **gen_kwargs):
        updated = 0
        log_key = self._to_log_key(input_data)

        llm_output = self.init_ans_model.generate(input_data, out_dict=True)
        llm_output['attempted_answer'] = llm_output.pop('init_answer').strip()
        llm_output['correct_answer'] = input_data['answer'] # used by verifier

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
                "examples": EXAMPLES_WORDSORT_UPDATE  # EXAMPLES_UPDATE
            }
            llm_output = chain_run_wrapper(self.chain_get_update, update_prompt)
            # check feedback
            prev_answer = llm_output['attempted_answer']
            llm_output['attempted_answer'] = llm_output.pop('updated_answer').strip()
            llm_output['correct_answer'] = input_data['answer']

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


class SelfImprove_GPT_WordSorting(SelfImprove_GPT_QA):
    def __init__(self, model, tokenizer, manual_prompt=False, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        super().__init__(model, tokenizer, manual_prompt, input_max_length, max_new_tokens, gen_kwargs=gen_kwargs)


TEMPLATE_GPTWORDSORT_UPDATE = """
{history}
Feedback: {feedback}
Updated Answer: Let's think step by step.
""".strip()

class GPTwVerifier_WordSorting(GenerativeModel):
    def __init__(
            self, 
            gpt_model: SelfImprove_woFeedback_GPT_QA, 
            verifier_model, 
            tokenizer=None,
            max_updates=10,
            verbose=False,
            save_log=True):
        self.gpt_model = gpt_model
        self.verifier_model = verifier_model  #.cuda()
        if not isinstance(self.verifier_model, WS_Verifier):
            # a neural verifier
            self.verifier_model.eval()
            self.tokenizer = tokenizer
        self.verbose = verbose
        self.save_log = save_log

        # other metadata
        self.num_updated = 0
        self.max_updates = max_updates
        self.logs: dict = {}
        return
    
    def _format_verifier_output(self, verifier_output):
        return verifier_output
    
    def get_verifier_feedback(self, llm_output):
        if isinstance(self.verifier_model, WS_Verifier):
            question = llm_output["question"]
            rationale = llm_output["attempted_answer"]
            target = llm_output['targets']
            try:
                cleaned_verifier_output = self.verifier_model.verify_rationale({
                    'rationale': rationale,
                    'question': question,
                    'target': target
                })
            except:
                cleaned_verifier_output = "[ERROR] The final response is also correct."
        else:
            verifier_prompt = {
                k: llm_output[k] for k in self.prompt_feedback.input_variables
            }
            verifier_input = self.prompt_feedback.format(**verifier_prompt)
            input_ids = self.tokenizer(
                verifier_input, 
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to("cuda")

            verifier_output = self.verifier_model.generate(input_ids, max_length=256, num_beams=1, num_return_sequences=1)
            decoded_verifier_output = self.tokenizer.decode(verifier_output[0], skip_special_tokens=True).strip()
            cleaned_verifier_output = self._format_verifier_output(decoded_verifier_output)
        return cleaned_verifier_output
    
    def check_if_done(self, feedback):
        return "final response is also correct" in feedback.lower()

    def _save_log(self, log_key, data):
        if log_key not in self.logs:
            self.logs[log_key] = []
        self.logs[log_key].append(data)
        return
    
    def _to_log_key(self, input_data):
        question = '\n'.join(input_data['inputs'].split('\n')[:-1])
        question = question.strip()
        formatted = f"""
        Q: {question}
        """.replace('    ', '').strip()
        return formatted
    
    def _extract_rationale(self, generated_attempt):
        all_steps = generated_attempt.split('\n')
        rationales = []
        for step in all_steps[::-1]:
            if "Feedback:" in step:
                break
            elif "Let's think step by step" in step:
                break
            rationales.insert(0, step)
        return '\n'.join(rationales).strip()
    
    def _get_improvement(self, history, generated_rationale, feedback):
        input_text = TEMPLATE_GPTWORDSORT_UPDATE.format(
            history=history,
            feedback=feedback
        ).strip() + "\n"
        generated_new_attempt = self.gpt_model.generate_until_done({
            'inputs': None,
            'data': {'history': input_text},
            'input_text': input_text,
            'attempted_answer': generated_rationale
        })
        improved_attempt = self._extract_rationale(generated_new_attempt)
        return generated_rationale, improved_attempt, generated_new_attempt

    def generate(self, input_data, **gen_kwargs):
        updated = 0
        log_key = self._to_log_key(input_data)

        generated_attempt = self.gpt_model.generate(input_data)
        generated_rationale = self._extract_rationale(generated_attempt)
        llm_output = {
            'question':  self._get_question(input_data),
            'attempted_answer': generated_rationale,
            'targets': input_data['targets'][0].split()
        }

        feedback = self.get_verifier_feedback(generated_rationale)
        if self.verbose:
            print("Feedback:", feedback)
        done = self.check_if_done(feedback)

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
            prev_answer, improved_answer, generated_new_attempt = self._get_improvement(generated_attempt, generated_rationale, feedback)
            # check feedback
            legit_update = self._check_attempted_answer_format(improved_answer, prev_answer)
            
            if legit_update:
                llm_output['targets'] = input_data['targets'][0].split()
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
            generated_rationale = improved_answer
            generated_attempt = generated_new_attempt
        return llm_output['attempted_answer'].strip()